from textwrap import dedent

from hotloop.agent.rules import apply


def write(tmp_path, source):
    (tmp_path / "train.py").write_text(dedent(source).lstrip())
    return tmp_path / "train.py"


def test_adds_workers_and_pinned_memory_without_reformatting(tmp_path):
    path = write(
        tmp_path,
        """
        loader = DataLoader(
            dataset,
            batch_size=64,  # keep this comment
            shuffle=True,
        )
    """,
    )
    assert apply(tmp_path) == ["train.py:4: DataLoader workers"]
    assert (
        path.read_text()
        == dedent("""
        loader = DataLoader(
            dataset,
            batch_size=64,  # keep this comment
            shuffle=True, num_workers=4, pin_memory=True,
        )
    """).lstrip()
    )


def test_replaces_num_workers_zero_and_leaves_set_values_alone(tmp_path):
    path = write(
        tmp_path,
        """
        a = DataLoader(ds, shuffle=True, num_workers=0, pin_memory=False)
        b = DataLoader(ds, shuffle=True, num_workers=args.workers)
    """,
    )
    apply(tmp_path)
    assert path.read_text().splitlines() == [
        "a = DataLoader(ds, shuffle=True, num_workers=4, pin_memory=False)",
        "b = DataLoader(ds, shuffle=True, num_workers=args.workers)",
    ]


def test_removes_syncs_from_training_loops_but_never_empties_a_block(tmp_path):
    path = write(
        tmp_path,
        """
        for x in loader:
            loss = model(x)
            loss.backward()
            torch.cuda.synchronize()
            if step % 100 == 0:
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
    """,
    )
    assert apply(tmp_path) == ["train.py:4: removed torch.cuda.synchronize()"]
    text = path.read_text()
    assert text.count("torch.cuda.synchronize()") == 1  # the one after the loop stays
    assert "        torch.cuda.empty_cache()" in text
    compile(text, "train.py", "exec")
