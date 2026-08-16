from hotloop.survey.scan import scan_repo


def test_reads_files_with_a_byte_order_mark(tmp_path):
    (tmp_path / "train.py").write_bytes(
        b"\xef\xbb\xbf"
        + b"import torch\n"
        + b"for x in loader:\n"
        + b"    with torch.autocast('cuda'):\n"
        + b"        loss = model(x)\n"
        + b"    loss.backward()\n"
    )
    found = scan_repo(tmp_path)
    assert found.training_loops == 1
    assert found.amp


def test_skips_test_files_and_finds_flags_in_scripts(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_loop.py").write_text("for x in d:\n    x.backward()\n")
    (tmp_path / "train.sh").write_text("python train.py --fp16\n")
    found = scan_repo(tmp_path)
    assert found.training_loops == 0
    assert found.amp == ["train.sh:1"]
