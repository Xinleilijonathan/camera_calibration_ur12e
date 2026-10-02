"""Segment splitting for scripts/replay_collect.py."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
replay_collect = importlib.import_module("replay_collect")

LIMIT = np.radians(30.0)


def test_small_move_is_one_segment():
    start = np.zeros(6)
    end = np.radians([10, -5, 0, 0, 20, 0])
    targets = replay_collect.split_transition(start, end, LIMIT)
    assert len(targets) == 1
    np.testing.assert_allclose(targets[-1], end)


def test_large_move_is_split_within_limit_and_ends_exactly():
    # camera_3 waypoint 28 -> 29: shoulder +63.7 deg
    start = np.radians([65.56, -86.08, -120.30, -112.20, 65.69, 16.07])
    end = np.radians([65.56, -22.37, -120.30, -112.20, 65.69, 16.07])
    targets = replay_collect.split_transition(start, end, LIMIT)
    assert len(targets) == 3
    previous = start
    for target in targets:
        assert np.max(np.abs(target - previous)) <= LIMIT + 1e-12
        previous = target
    np.testing.assert_array_equal(targets[-1], end)


def test_exact_multiple_of_limit_does_not_add_a_segment():
    start = np.zeros(6)
    end = np.zeros(6)
    end[0] = 2 * LIMIT
    assert len(replay_collect.split_transition(start, end, LIMIT)) == 2


def test_zero_move_returns_the_target():
    q = np.radians([1, 2, 3, 4, 5, 6])
    targets = replay_collect.split_transition(q, q, LIMIT)
    assert len(targets) == 1
    np.testing.assert_allclose(targets[0], q)


def test_non_positive_limit_is_rejected():
    with pytest.raises(ValueError):
        replay_collect.split_transition(np.zeros(6), np.ones(6), 0.0)
