from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _dates


class TestHMMExtension:

    def test_forward_filter_returns_correct_length(self):
        """Test _forward_filter with a mock model object."""
        from Strategy_Auto_Trader.extensions.hmm_extension import _forward_filter

        class MockModel:
            n_components = 3
            means_ = np.array([[0.01], [-0.01], [0.0]])
            covars_ = np.array([[[0.001]], [[0.001]], [[0.001]]])
            transmat_ = np.array([[0.7, 0.2, 0.1],
                                  [0.1, 0.7, 0.2],
                                  [0.2, 0.1, 0.7]])
            startprob_ = np.array([0.33, 0.34, 0.33])

        model = MockModel()
        X = np.random.randn(100, 1) * 0.01
        states = _forward_filter(model, X)
        assert len(states) == 100
        assert all(0 <= s < 3 for s in states)

    def test_forward_filter_deterministic(self):
        """Same input -> same output."""
        from Strategy_Auto_Trader.extensions.hmm_extension import _forward_filter

        class MockModel:
            n_components = 2
            means_ = np.array([[0.05], [-0.05]])
            covars_ = np.array([[[0.01]], [[0.01]]])
            transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
            startprob_ = np.array([0.5, 0.5])

        model = MockModel()
        np.random.seed(42)
        X = np.random.randn(50, 1) * 0.03
        s1 = _forward_filter(model, X)
        s2 = _forward_filter(model, X)
        np.testing.assert_array_equal(s1, s2)
