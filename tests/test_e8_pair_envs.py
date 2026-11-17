"""experiments/e8_pr61952/pair_envs.py on synthetic records with known answers: a PR
build that is faster by a known amount (F2 and F3 hold), one that is slower (F2 fails),
and one whose unit-16 cost is not smaller (F3a fails); plus its record loader, which
must read what the E2 judge reads."""
import json

from experiments.e8_pr61952 import pair_envs as p
from helpers import four_pass_records as make
from hplab.records import load_dir


def shift(cells, default=0.0, unit16=0.0):
    """The same requests with a constant added to the TTFT of each configuration."""
    out = {}
    for (block, ds, th), rows in cells.items():
        d = unit16 if block.startswith("C1") else default
        out[(block, ds, th)] = {k: dict(r, ttft_ms=r["ttft_ms"] + d) for k, r in rows.items()}
    return out


def test_known_faster_build_passes():
    main = make(overhead=33.0, noise=0.0)
    g = p.grid(main, shift(main, default=-12.0, unit16=-26.0))
    v = p.verdict(g)
    c = g["scored/smith/on"]
    assert c["default"]["median"] == -12.0 and c["unit16"]["median"] == -26.0
    assert c["cost_diff"]["median"] == -14.0 and c["cached_tokens_differ"] == 0
    assert v["F2_holds"] and v["F3a_holds"] and v["F3b_holds"]
    assert g["first/wildchat"]["default"]["n"] == 24  # thinking on and off pooled


def test_known_slower_build_fails_f2():
    main = make(overhead=33.0, noise=0.0)
    v = p.verdict(p.grid(main, shift(main, default=+90.0, unit16=+200.0)))
    assert not v["F2_holds"] and not v["F3a_holds"]


def test_cost_not_smaller_fails_f3a_only():
    main = make(overhead=33.0, noise=0.0)
    v = p.verdict(p.grid(main, shift(main, default=-12.0, unit16=-12.0)))
    assert v["F2_holds"] and not v["F3a_holds"] and v["F3b_holds"]


def test_request_missing_in_one_build_is_dropped():
    main = make(overhead=33.0, noise=0.0)
    pr = shift(main, default=-12.0, unit16=-26.0)
    key = sorted(pr[("C1b", "smith", "on")])[-1]
    del pr[("C1b", "smith", "on")][key]
    g = p.grid(main, pr)
    assert g["scored/smith/on"]["default"]["n"] == g["scored/smith/off"]["default"]["n"] - 1


def test_loader_reads_what_the_e2_judge_reads(tmp_path):
    rows = [{"sid": "control", "k": 0, "error": None}, {"sid": "a", "k": 0, "ttft_ms": 9.0, "error": None},
            {"sid": "a", "k": 1, "ttft_ms": None, "error": "boom"}]
    text = "".join(json.dumps(r) + "\n" for r in rows) + "\n"
    for name in ("C1a_smith_thinkon_R_s1_seed4.jsonl", "C0b_rebench_thinkoff_R_s1_seed5.jsonl",
                 "C0a_smith_thinkon_O_s1_seed4.jsonl", "C0a_smith_thinkon_R_s2_seed4.jsonl", "F0_x.jsonl"):
        (tmp_path / name).write_text(text, encoding="utf-8")
    got = p.load_dir(str(tmp_path))
    assert got == load_dir(str(tmp_path))
    assert sorted(got) == [("C0b", "rebench", "off"), ("C1a", "smith", "on")]
    assert list(got[("C1a", "smith", "on")]) == [(4, "a", 0)]
