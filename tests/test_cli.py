"""Smoke tests: every CLI command runs end-to-end on small inputs."""

import json

from reassemble.cli import main
from reassemble.model import save_case
from reassemble.synth.pages import render_page
from reassemble.synth.shredders import crosscut_shred, strip_shred, torn_shred


def test_synth_solve_audit_cli(tmp_path, capsys):
    assert main(["synth", "--kind", "memo", "-n", "12", "--out", str(tmp_path / "c")]) == 0
    assert main(["solve", str(tmp_path / "c"), "--out", str(tmp_path / "r"), "--scorer", "handcrafted"]) == 0
    res = json.loads((tmp_path / "r" / "result.json").read_text())
    assert res["metrics"]["neighbour_content"] > 0.8
    assert (tmp_path / "r" / "heatmap.png").exists()
    assert main(["audit", str(tmp_path / "c"), "--out", str(tmp_path / "a.pdf")]) == 0
    assert "strip-cut" in capsys.readouterr().out


def test_solve_2d_and_mat(tmp_path):
    page, _ = render_page("letter", 1)
    pcs, t = crosscut_shred(page, 3, 2, seed=1)
    save_case(tmp_path / "cc", pcs, t)
    assert main(["solve", str(tmp_path / "cc"), "--out", str(tmp_path / "rcc")]) == 0
    pcs, t = torn_shred(page, 2, 2, seed=1)
    save_case(tmp_path / "tn", pcs, t)
    assert main(["solve", str(tmp_path / "tn"), "--out", str(tmp_path / "rtn")]) == 0
    assert json.loads((tmp_path / "rtn" / "result.json").read_text())["metrics"]["neighbour"] == 1.0
    assert main(["mat", "--out", str(tmp_path / "mat.pdf")]) == 0
    assert (tmp_path / "mat.pdf").read_bytes()[:4] == b"%PDF"


def test_photo_capture_cli(tmp_path):
    assert main(["synth-photo", "--kind", "memo", "-n", "10", "--out", str(tmp_path / "ph")]) == 0
    assert main(["capture", str(tmp_path / "ph" / "photo.jpg"), "--out", str(tmp_path / "pc")]) == 0
    truth = json.loads((tmp_path / "pc" / "truth.json").read_text())
    assert len(truth["order"]) == 10 and not truth["missing"]


def test_bench_strips_smoke(capsys):
    from reassemble.bench import strip_benchmark
    from reassemble.pipeline import SolveOptions
    r = strip_benchmark(kinds=("memo",), n=12, seeds=range(1), opts=SolveOptions(scorer="handcrafted"))
    assert "all" in r and 0 <= r["all"]["neighbour"] <= 1
