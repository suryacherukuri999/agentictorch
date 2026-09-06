import pytest

from hotloop.scoreboard import outcome, recovered


def record(reason=None, measured=None, speedup=1.0):
    attempts = [{"change": "x", "keep": False, "reason": reason}] if reason else []
    return {"attempts": attempts, "measured_speedup": measured, "speedup": speedup}


def test_recovered_share_of_a_planted_slowdown():
    assert recovered(2.0, 2.0) == pytest.approx(1.0)
    assert recovered(1.5, 2.0) == pytest.approx((1 - 1 / 1.5) / 0.5)
    assert recovered(1.0, 1.0) is None


def test_blind_patches_ship_whatever_they_do():
    assert outcome(record("run failed: SyntaxError"), "llm") == (1.0, True)
    assert outcome(record("loss moved by 0.5, more than 0.1", 2.0), "llm-kb") == (1.0, True)
    assert outcome(record("0.80x is within noise", 0.8), "llm") == (0.8, True)
    assert outcome(record("1.01x is within noise", 1.01), "rules") == (1.01, False)
    assert outcome(record(), "rules") == (1.0, False)


def test_the_agent_ships_only_what_it_kept():
    assert outcome(record("run failed: SyntaxError", speedup=2.5), "agent") == (2.5, False)
