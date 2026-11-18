"""analyze_e10.py on hand-made client records and hook events (no vLLM needed)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "e10_state_eviction"))
import analyze_e10 as A  # noqa: E402

KINDS = ["attn", "mamba", "mamba", "mamba"]


def _write(d, label, rows, events):
    os.makedirs(os.path.join(d, "hook"), exist_ok=True)
    with open(os.path.join(d, f"{label}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"label": label, **r}) + "\n")
    with open(os.path.join(d, "hook", "events_1.jsonl"), "w") as f:
        for e in [{"ev": "init", "t": 1, "groups": KINDS}] + events:
            f.write(json.dumps(e) + "\n")


def _case(tmp_path):
    """Two sessions, three turns each (P = 2000, 2250, 2500). Session 0 hits every turn;
    session 1 misses turn 1 with only the state gone and turn 2 with both gone."""
    rows, events, t = [], [], 100
    hits = {(0, 1): 1984, (0, 2): 2240, (1, 1): 0, (1, 2): 0}
    groups = {(0, 1): [2240, 1984, 1984, 1984], (0, 2): [2496, 2240, 2240, 2240],
              (1, 1): [2240, 0, 0, 0], (1, 2): [0, 0, 0, 0]}
    for k in range(3):
        for s in range(2):
            P = 2000 + 250 * k
            rid = f"r{s}{k}"
            hit = hits.get((s, k), 0)
            rows.append({"sid": f"synth-{s}", "k": k, "P": P, "pred_inf": A.checkpoint(P - 250) if k else 0,
                         "cached_tokens": hit, "t_send_ns": t, "t_end_ns": t + 50})
            # a lookup while waiting (no hit), then the admission lookup
            events.append({"ev": "lookup", "t": t + 1, "rid": rid, "P": P, "tok0": 500 + s, "hit": 0,
                           "groups": [0, 0, 0, 0], "preempted": 0})
            events.append({"ev": "lookup", "t": t + 2, "rid": rid, "P": P, "tok0": 500 + s, "hit": hit,
                           "groups": groups.get((s, k), [0, 0, 0, 0]), "preempted": 0})
            ref = 1 if s == 0 else 0
            events.append({"ev": "end", "t": t + 40, "rid": rid, "P": P, "c": A.checkpoint(P),
                           "state": [{"g": g, "present": True, "ref": ref} for g in (1, 2, 3)]})
            t += 100
    events.append({"ev": "evict", "t": 150, "g": 1, "kind": "mamba", "alloc": True})
    events.append({"ev": "evict", "t": 160, "g": 0, "kind": "attn", "alloc": False})
    rows.append({"sid": "control", "control": "B", "k": 0, "P": 1500, "cached_tokens": 0,
                 "t_send_ns": 10, "ttft_ms": 0.00002, "t_end_ns": 30})
    events.append({"ev": "lookup", "t": 20, "rid": "cb", "P": 1500, "tok0": 1, "hit": 0,
                   "groups": [0, 0, 0, 0], "preempted": 0})
    rows.append({"sid": "control", "control": "A_again", "k": 0, "P": 1500, "cached_tokens": 1488,
                 "t_send_ns": 40, "ttft_ms": 0.00002, "t_end_ns": 70})
    events.append({"ev": "lookup", "t": 50, "rid": "ca", "P": 1500, "tok0": 1, "hit": 1488,
                   "groups": [1488, 1488, 1488, 1488], "preempted": 0})
    events.sort(key=lambda e: e["t"])
    _write(str(tmp_path), "C1_S_synth_kv3456_out16_seed1", rows, events)
    return str(tmp_path)


def test_classify():
    assert A.classify(2240, 0, 1984) == "state_only"
    assert A.classify(0, 0, 1984) == "both"
    assert A.classify(0, 1984, 1984) == "attn_only"


def test_file_and_cell(tmp_path):
    d = _case(tmp_path)
    clients, events = A.load_dir(d)
    kinds, _ = A.group_kinds(events)
    (label, rows), = clients.items()
    f = A.analyze_file(label, rows, events, kinds, 2000, 250)
    assert f["n_sessions_seen"] == 2 and f["matched"] == 6 and f["agree"] == 6
    assert f["n"] == 4 and f["zero"] == 2
    assert f["classes"] == {"state_only": 1, "both": 1}
    assert f["hit"] == 1984 + 2240 and f["pred"] == 2 * (1984 + 2240)
    assert f["end"] == {"held": 3, "free_queue": 3}
    assert f["evict"] == {"mamba": 1}
    cell = A.summarize([f])["3456MiB_out16"]
    assert cell["retention"] == 0.5 and cell["share_state_only"] == 0.5 and cell["end_not_held"] == 0.5
    assert cell["controls_ok"] == [True]


def test_control_check_catches_a_wrong_classifier():
    good = {"B": {"hit": 0, "client": 0, "attn": 0, "state": 0},
            "A_again": {"hit": 1488, "client": 1488, "attn": 1488, "state": 1488}}
    assert A.control_check(good) is True
    bad = dict(good, B={"hit": 0, "client": 0, "attn": 1488, "state": 0})
    assert A.control_check(bad) is False
    bad = dict(good, A_again={"hit": 1488, "client": 1472, "attn": 1488, "state": 1488})
    assert A.control_check(bad) is False


def test_sessions_spelled_in_two_tokens(tmp_path):
    """Fewer distinct vocabulary ids (2) than sessions (4): the session index is spelled in
    the first two tokens, as pressure_sessions.synth_sessions does."""
    vocab = [40, 90]
    rows, events, t = [], [], 100
    for s in range(4):
        head = [vocab[s % 2], vocab[s // 2 % 2], 3, 3]
        for k in range(2):
            P = 2000 + 250 * k
            rows.append({"sid": f"synth-{s}", "k": k, "P": P, "pred_inf": A.checkpoint(P - 250) if k else 0,
                         "cached_tokens": 100 * s + k, "t_send_ns": t, "t_end_ns": t + 50})
            events.append({"ev": "lookup", "t": t + 1, "rid": f"r{s}{k}", "P": P, "tok0": head[0], "head": head,
                           "hit": 100 * s + k, "groups": [0, 0, 0, 0], "preempted": 0})
            t += 100
    _write(str(tmp_path), "C1_S_synth_kv5184_out16_seed2", rows, events)
    clients, ev = A.load_dir(str(tmp_path))
    (label, r), = clients.items()
    f = A.analyze_file(label, r, ev, KINDS, 2000, 250)
    assert f["matched"] == 8 and f["agree"] == 8 and f["n_sessions_seen"] == 4


def test_cell_names():
    assert A.cell_of("C1_S_synth_kv5184_out256_seed3") == "5184MiB_out256"
    assert A.cell_of("cpu_kv288b_out16_seed1_gap1") == "288blocks_out16"
    assert A.cell_of("C1_L_smith_kv3330_out16_seed2") == "3330MiB_out16"


def test_recorded_sessions_matched_by_window(tmp_path):
    """Recorded sessions carry no session id in their tokens: each request is matched to the
    one lookup id that starts inside its send..end window with the same prompt length.
    Request a1 overlaps b1 in time but differs in length; c1 has two candidates (same
    length, both inside its window) and stays unmatched. a1 stops short at an earlier
    checkpoint because only the state is missing, b1 misses with both gone."""
    rows = [
        {"sid": "s-a", "k": 0, "P": 8000, "pred_inf": 0, "cached_tokens": 0, "t_send_ns": 100, "t_end_ns": 200},
        {"sid": "s-a", "k": 1, "P": 9000, "pred_inf": 7984, "cached_tokens": 4592,
         "t_send_ns": 300, "t_end_ns": 500},
        {"sid": "s-b", "k": 1, "P": 9100, "pred_inf": 8000, "cached_tokens": 0, "t_send_ns": 310, "t_end_ns": 520},
        {"sid": "s-c", "k": 1, "P": 7000, "pred_inf": 6000, "cached_tokens": 6000, "t_send_ns": 600, "t_end_ns": 700},
    ]
    events = [
        {"ev": "lookup", "t": 101, "rid": "x0", "P": 8000, "tok0": 1, "hit": 0, "groups": [0, 0, 0, 0]},
        {"ev": "lookup", "t": 305, "rid": "x1", "P": 9000, "tok0": 1, "hit": 4592,
         "groups": [8640, 4592, 4592, 4592], "preempted": 0},
        {"ev": "lookup", "t": 312, "rid": "x2", "P": 9100, "tok0": 1, "hit": 0, "groups": [576, 0, 0, 0],
         "preempted": 0},
        {"ev": "lookup", "t": 601, "rid": "x3", "P": 7000, "tok0": 1, "hit": 6000, "groups": [6000] * 4},
        {"ev": "lookup", "t": 602, "rid": "x4", "P": 7000, "tok0": 1, "hit": 6000, "groups": [6000] * 4},
    ]
    _write(str(tmp_path), "C1_L_smith_kv3330_out16_seed1", rows, events)
    clients, ev = A.load_dir(str(tmp_path))
    (label, r), = clients.items()
    f = A.analyze_file(label, r, ev, KINDS, 2000, 250)
    assert f["matched"] == 3 and f["agree"] == 3 and f["ambiguous"] == 1
    assert f["n"] == 3 and f["zero"] == 1 and f["classes"] == {"both": 1}
    assert f["short"] == 2 and f["short_classes"] == {"state_only": 1, "both": 1}
