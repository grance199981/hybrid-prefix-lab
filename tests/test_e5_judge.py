"""experiments/e5_cache_pressure/pressure_judge.py on synthetic records with known
answers: a grid where the hybrid model keeps up (the kill criterion must hold), one
where it falls far behind (must not hold), failing and missing controls (no
verdict), and the pieces (classify, steady_window, reconciliation, compare_sim,
file loading)."""
import json
import os
import sys

import pytest

# the experiment's modules import each other by plain name, as when run as scripts
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "e5_cache_pressure"))
import pressure_judge as j  # noqa: E402

P, PRED = 2000, 1600


def window_order(n_sessions, turns, live):
    """[(session, k)]: `live` sessions take turns; a finished one is replaced by the next."""
    order, active, nxt = [], [[s, 0] for s in range(min(live, n_sessions))], live
    while active:
        for slot in list(active):
            order.append(tuple(slot))
            slot[1] += 1
            if slot[1] == turns:
                if nxt < n_sessions:
                    slot[0], slot[1], nxt = nxt, 0, nxt + 1
                else:
                    active.remove(slot)
    return order


def run(keep, seed=1, preemptions=0.0, n_sessions=12, turns=5, live=4, reconciles=True):
    """One window-mode run. keep: share of the requests with k > 0 that hit pred_inf,
    spread evenly over the run; the others hit 0. With the defaults the steady
    window is the 32 requests with k > 0 of sessions 0..7 (session 11 starts at seq 43)."""
    rows, later = [], 0
    for seq, (s, k) in enumerate(window_order(n_sessions, turns, live)):
        hit = k > 0 and int((later + 1) * keep) > int(later * keep)
        later += k > 0
        rows.append({"sid": f"s{s}", "k": k, "seq": seq, "P": P, "n_out": 16, "worker": None,
                     "prompt_tokens": P, "pred_inf": PRED if k > 0 else 0,
                     "cached_tokens": PRED if hit else 0, "ttft_ms": 20.0, "error": None})
    cached = sum(r["cached_tokens"] for r in rows)
    summary = {"sum_prompt_tokens": P * len(rows), "sum_cached_tokens": cached,
               "cross_prefix": {f"s{s}": 0 for s in range(n_sessions)},
               "delta": {j.PREEMPT: preemptions, j.QUERIES: float(P * len(rows)),
                         j.HITS: float(cached if reconciles else cached + 64), j.EVICT: 5.0}}
    return {"seed": seed, "rows": rows, "summary": summary, "metrics": [{j.USAGE: 0.4}, {j.USAGE: 0.97}]}


def grid(f0=0.8, c1=0.8, override=None, drop=(), controls=True, **kw):
    """Stage S: pools 1024 / 2048 / 4096 MiB x outputs 16 / 256, synthetic sessions.
    Stage I: the same pools, output 64, SWE-smith. Stage K: an ample and a minimum pool."""
    cells = {}
    for cfg, keep in (("F0", f0), ("C1", c1)):
        for kv in (1024, 2048, 4096):
            for stage, wl, outs in (("S", "synth", (16, 256)), ("I", "smith", (64,))):
                for o in outs:
                    key = (cfg, stage, wl, kv, o)
                    if key not in drop:
                        cells[key] = [run((override or {}).get(key, keep), seed, **kw) for seed in (1, 2)]
        if controls:
            cells[(cfg, "K", "synth", 30000, 16)] = [run(1.0)]
            cells[(cfg, "K", "synth", 256, 16)] = [run(0.1)]
    return cells


# ---------------------------------------------------------------------- pieces
def test_classify_each_class():
    row = lambda c: {"cached_tokens": c, "pred_inf": 1600}      # noqa: E731
    assert j.classify(row(1600), 128) == "full"
    assert j.classify(row(1024), 128) == "older"         # an older checkpoint / part of the chain
    assert j.classify(row(128), 128) == "shared_only"    # no more than other sessions share too
    assert j.classify(row(64), 128) == "shared_only"
    assert j.classify(row(0), 128) == "zero"
    assert j.classify(row(1664), 128) == "above"
    assert j.classify({"cached_tokens": 0, "pred_inf": 0}, 0) == "full"


def test_retention_and_cell_stats():
    s = j.cell_stats([run(0.75), run(0.25, seed=2, preemptions=3.0)])
    assert s["n"] == 2 * 32
    assert s["retention"] == 0.5             # 24 of 32 and 8 of 32
    assert s["class_share"]["full"] + s["class_share"]["zero"] == pytest.approx(1.0)
    assert s["recompute_median"] == P - PRED / 2 and s["recompute_p90"] == P    # half hit, half miss
    assert s["preemptions"] == 3.0 and s["evictions"] == 10.0 and s["kv_usage_max"] == 0.97
    assert j.retention([]) is None and j.cell_stats([{**run(1.0), "rows": []}]) is None


def test_steady_window_window_mode():
    rows = run(1.0, n_sessions=4, turns=4)["rows"]        # seq 0..15, last session starts at seq 3
    assert j.steady_window(rows, 0) == []                 # every k > 0 comes after the last start
    # sessions s2, s3 start late (their k = 0 at seq 8, 9): the live count is full until then
    late = [dict(r) for r in rows]
    order = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
    for seq, i in enumerate(order):
        late[i]["seq"] = seq
    kept = j.steady_window(late, 0)
    assert sorted(r["seq"] for r in kept) == [2, 3, 4, 5, 6, 7]
    # warm-up: requests count only after the earlier ones recomputed pool_tokens.
    # seq 0, 1 recompute 2000 each, seq 2.. recompute 400 each: 4800 is reached after seq 3
    assert sorted(r["seq"] for r in j.steady_window(late, 4800)) == [4, 5, 6, 7]
    assert j.steady_window(late, 10**9) == []
    errors = [dict(r, error="boom", cached_tokens=None) if r["seq"] == 5 else r for r in late]
    assert sorted(r["seq"] for r in j.steady_window(errors, 0)) == [2, 3, 4, 6, 7]


def test_steady_window_closed_mode_drops_each_worker_s_first_session():
    rows, seq = [], 0
    for sid, w in (("a", 0), ("b", 1), ("c", 0), ("d", 1)):
        for k in range(3):
            rows.append({"sid": sid, "k": k, "seq": seq, "worker": w, "P": 100, "cached_tokens": 0,
                         "pred_inf": 64, "error": None})
            seq += 1
    kept = j.steady_window(rows, 10**9)                   # the token cut does not apply
    assert [(r["sid"], r["k"]) for r in kept] == [("c", 1), ("c", 2), ("d", 1), ("d", 2)]


def test_steady_window_closed_mode_one_session_per_worker_keeps_every_later_turn():
    """Stage S: each worker runs a single session, so there is no warm-up session to cut.
    Cutting each worker's first session here would remove every request."""
    rows, seq = [], 0
    for sid, w in (("a", 0), ("b", 1), ("c", 2)):
        for k in range(3):
            rows.append({"sid": sid, "k": k, "seq": seq, "worker": w, "P": 100, "cached_tokens": 0,
                         "pred_inf": 64, "error": "boom" if (sid, k) == ("b", 2) else None})
            seq += 1
    kept = j.steady_window(rows, 10**9)
    assert [(r["sid"], r["k"]) for r in kept] == [("a", 1), ("a", 2), ("b", 1), ("c", 1), ("c", 2)]
    runs = [{"seed": 1, "rows": rows, "summary": {}, "metrics": []}]
    assert j.cell_stats(runs)["n"] == 5


def test_compare_sim_per_rule_set():
    rows = [{"sid": "a", "k": 0, "cached_tokens": 0, "error": None},
            {"sid": "a", "k": 1, "cached_tokens": 1152, "error": None},
            {"sid": "a", "k": 2, "cached_tokens": 0, "error": None},
            {"sid": "a", "k": 3, "cached_tokens": None, "error": "boom"},
            {"sid": "control", "k": 0, "cached_tokens": 0, "error": None}]
    sim = lambda hits: [{"sid": "a", "k": k, "hit": h} for k, h in enumerate(hits)]      # noqa: E731
    out = j.compare_sim(rows, {"M0": sim([0, 1152, 1728, 0]), "M1": sim([0, 1152, 0, 0])})
    assert out == {"M0": pytest.approx(2 / 3), "M1": 1.0, "n": 3}


# ------------------------------------------------------------------- verdicts
def test_known_good_grid_holds():
    res = j.judge(grid(f0=0.8, c1=0.8))
    assert res["controls"]["problems"] == []
    assert [c["holds"] for c in res["controls"]["ample"]] == [True, True]
    assert [c["holds"] for c in res["controls"]["minimum"]] == [True, True]
    assert all(c["holds"] for c in res["controls"]["reconciliation"]) and res["controls"]["reconciliation"]
    kc = res["kill_criterion"]
    assert kc["complete"] and kc["no_concurrency_specific_problem"] is True
    # the two tightest pools (not 4096), both output lengths; I: two pools, one length
    assert [(r["F0_kv"], r["C1_kv"], r["out"]) for r in kc["S"]] == [(1024, 1024, 16), (1024, 1024, 256),
                                                                    (2048, 2048, 16), (2048, 2048, 256)]
    assert len(kc["I"]) == 2 and all(r["ratio"] == 1.0 for r in kc["I"])
    assert len(kc["same_bytes"]) == 9 and all(r["holds"] for r in kc["S"] + kc["I"] + kc["same_bytes"])


def test_known_bad_grid_does_not_hold():
    res = j.judge(grid(f0=0.8, c1=0.3))
    assert res["controls"]["problems"] == []
    kc = res["kill_criterion"]
    assert kc["complete"] and kc["no_concurrency_specific_problem"] is False
    assert not any(r["holds"] for r in kc["S"] + kc["I"] + kc["same_bytes"])
    assert kc["S"][0]["retention_F0"] - kc["S"][0]["retention_C1"] > 0.4
    assert kc["I"][0]["ratio"] > 1.5              # C1 recomputes the whole prompt, F0 a fifth of it


def test_each_condition_can_fail_on_its_own():
    # a single S cell 15 points behind
    kc = j.judge(grid(c1=0.85, override={("C1", "S", "synth", 2048, 256): 0.62}))["kill_criterion"]
    assert kc["no_concurrency_specific_problem"] is False
    assert [r["holds"] for r in kc["S"]] == [True, True, True, False] and all(r["holds"] for r in kc["I"])
    # 5 points behind passes (S) and (I) but not "same bytes"
    kc = j.judge(grid(f0=0.8, c1=0.75))["kill_criterion"]
    assert all(r["holds"] for r in kc["S"] + kc["I"]) and not any(r["holds"] for r in kc["same_bytes"])
    assert kc["no_concurrency_specific_problem"] is False
    # only the largest pool behind: not among the two tightest, caught by "same bytes" only
    kc = j.judge(grid(override={("C1", "S", "synth", 4096, 16): 0.2}))["kill_criterion"]
    assert all(r["holds"] for r in kc["S"] + kc["I"])
    assert [r["holds"] for r in kc["same_bytes"]].count(False) == 1


def test_more_preemptions_in_the_hybrid_model_fail_s():
    cells = grid()
    cells[("C1", "S", "synth", 1024, 16)] = [run(0.8, preemptions=2.0)]
    res = j.judge(cells)
    assert res["controls"]["problems"] == []      # the run with preemptions is not reconciled
    assert [c["run"][:5] for c in res["controls"]["reconciliation_skipped"]] == [("C1", "S", "synth", 1024, 16)]
    kc = res["kill_criterion"]
    assert kc["S"][0]["holds"] is False and kc["S"][0]["preemptions_C1"] == 2.0
    assert kc["no_concurrency_specific_problem"] is False


def test_missing_cell_gives_no_verdict():
    kc = j.judge(grid(drop={("C1", "S", "synth", 1024, 256)}))["kill_criterion"]
    assert kc["no_concurrency_specific_problem"] is None and not kc["complete"]
    assert any("a cell is missing" in m for m in kc["missing"])
    kc = j.judge(grid(drop={(c, "I", "smith", kv, 64) for c in ("F0", "C1")
                            for kv in (1024, 2048, 4096)}))["kill_criterion"]
    assert kc["no_concurrency_specific_problem"] is None and "stage I: no cells" in kc["missing"]


def test_failing_control_withholds_the_verdict():
    # ample cell loses hits: the client, the server or pred_inf is off, so nothing can be read
    cells = grid()
    cells[("C1", "K", "synth", 30000, 16)] = [run(0.9)]
    kc = j.judge(cells)["kill_criterion"]
    assert kc["no_concurrency_specific_problem"] is None and kc["verdict_withheld"] is True
    assert "ample control" in kc["reason"]
    # minimum cell loses nothing: the measurement cannot see a loss
    cells = grid()
    cells[("F0", "K", "synth", 256, 16)] = [run(0.9)]
    kc = j.judge(cells)["kill_criterion"]
    assert kc["no_concurrency_specific_problem"] is None and "minimum control" in kc["reason"]
    # no control cells at all
    kc = j.judge(grid(controls=False))["kill_criterion"]
    assert kc["no_concurrency_specific_problem"] is None and "stage K" in kc["reason"]


def test_reconciliation_is_checked_only_without_preemptions():
    bad = j.judge(grid(reconciles=False))
    assert bad["kill_criterion"]["no_concurrency_specific_problem"] is None
    assert "reconciliation" in bad["kill_criterion"]["reason"]
    assert [c["holds"] for c in bad["controls"]["reconciliation"]].count(False) == 2 * 9 * 2
    skipped = j.judge(grid(reconciles=False, preemptions=1.0))
    assert skipped["controls"]["problems"] == [] and skipped["controls"]["reconciliation"] == [
        {"run": ("C1", "K", "synth", 256, 16, 1), "holds": True},
        {"run": ("C1", "K", "synth", 30000, 16, 1), "holds": True},
        {"run": ("F0", "K", "synth", 256, 16, 1), "holds": True},
        {"run": ("F0", "K", "synth", 30000, 16, 1), "holds": True}]
    assert len(skipped["controls"]["reconciliation_skipped"]) == 2 * 9 * 2
    assert skipped["kill_criterion"]["no_concurrency_specific_problem"] is True


# ----------------------------------------------------------------------- files
def test_load_dir_reads_tags_and_main_prints_controls_before_the_verdict(tmp_path, capsys):
    for key, runs in grid().items():
        for r in runs:
            stem = tmp_path / "{}_{}_{}_kv{}_out{}_seed{}".format(*key, r["seed"])
            control = {"sid": "control", "control": "A", "k": 0, "P": 1500, "cached_tokens": 0, "error": None}
            with open(f"{stem}.jsonl", "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(x) + "\n" for x in [control] + r["rows"]))
            with open(f"{stem}.summary.json", "w", encoding="utf-8") as f:
                json.dump(r["summary"], f)
            with open(f"{stem}.metrics.jsonl", "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(x) + "\n" for x in r["metrics"]))
    (tmp_path / "C1_S_synth_kv512_out16_seed1.jsonl").write_text("", encoding="utf-8")   # no summary: stopped run
    (tmp_path / "client_F0_kv1024.jsonl").write_text("", encoding="utf-8")               # not a tag
    cells = j.load_dir(str(tmp_path))
    assert set(cells) == set(grid()) and len(cells[("F0", "S", "synth", 1024, 16)]) == 2
    assert j.judge(cells)["kill_criterion"]["no_concurrency_specific_problem"] is True
    out_json = tmp_path / "verdict.json"
    assert j.main([str(tmp_path), "--tokens-per-mib", "F0=0.5", "C1=0.5", "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert 0 < text.index("== controls") < text.index("== kill criterion") \
        < text.index("== no_concurrency_specific_problem: True")
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["kill_criterion"]["no_concurrency_specific_problem"] is True
    assert "F0/S/synth/1024/16" in saved["cells"]
