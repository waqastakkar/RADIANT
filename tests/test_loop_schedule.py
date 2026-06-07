import pytest

from training import (
    CurriculumLoopSchedule,
    FixedLoopSchedule,
    UniformLoopSchedule,
)


def test_fixed_returns_constant():
    s = FixedLoopSchedule(3)
    for step in range(10):
        assert s.sample(step) == 3


def test_fixed_invalid_n():
    with pytest.raises(ValueError):
        FixedLoopSchedule(0)


def test_uniform_within_bounds():
    s = UniformLoopSchedule(low=2, high=5, seed=7)
    seen = {s.sample(i) for i in range(50)}
    assert seen.issubset({2, 3, 4, 5})


def test_curriculum_anneals():
    s = CurriculumLoopSchedule(start=1, end=8, n_steps=10)
    assert s.sample(0) == 1
    assert s.sample(10) == 8
    assert s.sample(100) == 8  # clamps at end


def test_curriculum_monotonic():
    s = CurriculumLoopSchedule(start=2, end=10, n_steps=8)
    samples = [s.sample(i) for i in range(9)]
    for a, b in zip(samples, samples[1:]):
        assert a <= b
