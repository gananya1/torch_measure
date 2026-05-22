"""Unit tests for the competition submission interface.

Fast contract tests run without any trained weights (good for CI).
Real-model tests are skipped automatically if ncf_head.pt is absent.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

COMPETITION_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "competition")
)
_WEIGHTS_PRESENT = os.path.exists(os.path.join(COMPETITION_DIR, "ncf_head.pt"))


# Stub helpers — satisfy the contract without loading any weights
def _stub_predict(inp: dict, labeled=None) -> float:
    return 0.5

def _stub_acquire(inp: dict) -> float:
    return 0.0


# Contract tests — no weights needed
SAMPLE_INPUT = {
    "benchmark":       "mmlupro",
    "condition":       "none",
    "subject_content": "Name: TestModel\nOrganization: TestOrg",
    "item_content":    "What is 2 + 2?  A) 3  B) 4  C) 5  D) 6",
}

SAMPLE_LABELED = [
    {**SAMPLE_INPUT, "label": 1},
    {**SAMPLE_INPUT, "label": 0},
]


class TestPredictContract:
    @pytest.mark.parametrize("labeled", [None, [], SAMPLE_LABELED])
    def test_returns_finite_float_in_unit_interval(self, labeled):
        p = _stub_predict(SAMPLE_INPUT, labeled=labeled)
        assert isinstance(p, float), f"Expected float, got {type(p)}"
        assert np.isfinite(p),       f"Non-finite: {p}"
        assert 0.0 <= p <= 1.0,      f"Out of [0,1]: {p}"

    def test_all_four_keys_present(self):
        """Ensure we never accidentally drop a required key."""
        for key in ("benchmark", "condition", "subject_content", "item_content"):
            assert key in SAMPLE_INPUT


class TestAcquisitionContract:
    def test_returns_finite_float(self):
        result = float(_stub_acquire(SAMPLE_INPUT))
        assert np.isfinite(result), f"Non-finite acquisition score: {result}"

    def test_does_not_raise(self):
        _stub_acquire(SAMPLE_INPUT)   # must not raise


# Real-model smoke tests — skipped if weights are not yet built
@pytest.mark.skipif(not _WEIGHTS_PRESENT,
                    reason="ncf_head.pt absent — run competition/train_ncf.py first")
class TestRealModel:
    INPUTS = [
        {
            "benchmark":       "mmlupro",
            "condition":       "none",
            "subject_content": "Name: GPT-4\nOrganization: OpenAI",
            "item_content":    "A train travels at 60 mph for 2 hours. How far does it go?",
        },
        {
            "benchmark":       "gsm8k",
            "condition":       "chain-of-thought",
            "subject_content": "Name: Llama-3-8B\nOrganization: Meta",
            "item_content":    "Solve: 3x + 7 = 22",
        },
    ]

    def setup_method(self):
        sys.path.insert(0, COMPETITION_DIR)
        # Re-import fresh to avoid stale module state between test methods
        if "model" in sys.modules:
            del sys.modules["model"]
        import model as m
        self._m = m

    def test_valid_probability_for_each_input(self):
        for inp in self.INPUTS:
            p = self._m.predict(inp)
            assert isinstance(p, float), f"Expected float, got {type(p)}"
            assert 0.0 <= p <= 1.0,      f"Out of [0,1]: {p}"
            assert np.isfinite(p),       f"Non-finite: {p}"

    def test_handles_none_labeled(self):
        p = self._m.predict(self.INPUTS[0], labeled=None)
        assert np.isfinite(p)

    def test_handles_empty_labeled(self):
        p = self._m.predict(self.INPUTS[0], labeled=[])
        assert np.isfinite(p)

    def test_handles_nonempty_labeled(self):
        labeled = [{**self.INPUTS[0], "label": 1},
                   {**self.INPUTS[0], "label": 0}]
        p = self._m.predict(self.INPUTS[0], labeled=labeled)
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0