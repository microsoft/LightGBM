# coding: utf-8
"""
Acceptance tests for MoE (Mixture-of-Experts) extension.
Based on specification section 8:
- toy data training with mixture_enable=1, K=2 should not produce NaN
- predict_regime_proba rows should sum to 1 (tolerance 1e-6)
- save->load should produce consistent predict / predict_regime_proba
- mixture_enable=0 should match standard GBDT behavior
- Python import works
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

import lightgbm_moe as lgb


def make_toy_regression_data(n_samples=500, n_features=10, n_regimes=2, random_state=42):
    """Generate toy regression data with underlying regimes."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)

    # Create regime-based y
    regime = (X[:, 0] > 0).astype(int)  # Simple regime based on first feature
    y = np.zeros(n_samples)

    # Regime 0: y = 2*x1 + noise
    mask0 = regime == 0
    y[mask0] = 2 * X[mask0, 1] + rng.randn(mask0.sum()) * 0.5

    # Regime 1: y = -3*x2 + 5 + noise
    mask1 = regime == 1
    y[mask1] = -3 * X[mask1, 2] + 5 + rng.randn(mask1.sum()) * 0.5

    return X, y, regime


class TestMixtureImport:
    """Test that the rebranded package imports correctly."""

    def test_import_lightgbm_moe(self):
        """Test that import lightgbm_moe works."""
        import lightgbm_moe
        assert hasattr(lightgbm_moe, 'Dataset')
        assert hasattr(lightgbm_moe, 'train')
        assert hasattr(lightgbm_moe, 'Booster')

    def test_basic_train_predict(self):
        """Test basic train/predict works (non-MoE)."""
        X, y, _ = make_toy_regression_data(n_samples=100)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
        }
        bst = lgb.train(params, train_data, num_boost_round=10)
        pred = bst.predict(X)

        assert pred.shape == (100,)
        assert not np.any(np.isnan(pred))


class TestMixtureTraining:
    """Test MoE training functionality."""

    def test_mixture_training_no_nan(self):
        """Toy data training with mixture_enable=1, K=2 should not produce NaN."""
        X, y, _ = make_toy_regression_data(n_samples=200, n_regimes=2)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'boosting': 'mixture',
            'mixture_enable': True,
            'mixture_num_experts': 2,
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
        }

        bst = lgb.train(params, train_data, num_boost_round=20)
        pred = bst.predict(X)

        # Check no NaN in predictions
        assert not np.any(np.isnan(pred)), "Predictions contain NaN"
        assert pred.shape == (200,)

    def test_mixture_training_k4(self):
        """Test MoE training with K=4 experts."""
        X, y, _ = make_toy_regression_data(n_samples=300)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'boosting': 'mixture',
            'mixture_enable': True,
            'mixture_num_experts': 4,
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
        }

        bst = lgb.train(params, train_data, num_boost_round=15)
        pred = bst.predict(X)

        assert not np.any(np.isnan(pred))
        assert bst.is_mixture()
        assert bst.num_experts() == 4


class TestMixturePrediction:
    """Test MoE prediction functionality."""

    @pytest.fixture
    def trained_mixture_model(self):
        """Create a trained MoE model for prediction tests."""
        X, y, _ = make_toy_regression_data(n_samples=200)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'boosting': 'mixture',
            'mixture_enable': True,
            'mixture_num_experts': 3,
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
        }

        bst = lgb.train(params, train_data, num_boost_round=20)
        return bst, X

    def test_predict_regime_proba_sum_to_one(self, trained_mixture_model):
        """predict_regime_proba rows should sum to 1 (tolerance 1e-6)."""
        bst, X = trained_mixture_model
        regime_proba = bst.predict_regime_proba(X)

        # Check shape
        assert regime_proba.shape == (200, 3), f"Expected shape (200, 3), got {regime_proba.shape}"

        # Check rows sum to 1
        row_sums = regime_proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6,
                                   err_msg="regime_proba rows do not sum to 1")

        # Check all values are in [0, 1]
        assert np.all(regime_proba >= 0), "regime_proba contains negative values"
        assert np.all(regime_proba <= 1), "regime_proba contains values > 1"

    def test_predict_regime_returns_valid_indices(self, trained_mixture_model):
        """predict_regime should return valid expert indices."""
        bst, X = trained_mixture_model
        regimes = bst.predict_regime(X)

        # Check shape
        assert regimes.shape == (200,), f"Expected shape (200,), got {regimes.shape}"

        # Check values are valid indices
        assert np.all(regimes >= 0), "regime contains negative values"
        assert np.all(regimes < 3), "regime contains values >= num_experts"

    def test_predict_expert_pred_shape(self, trained_mixture_model):
        """predict_expert_pred should return (n_samples, n_experts) array."""
        bst, X = trained_mixture_model
        expert_preds = bst.predict_expert_pred(X)

        # Check shape
        assert expert_preds.shape == (200, 3), f"Expected shape (200, 3), got {expert_preds.shape}"

        # Check no NaN
        assert not np.any(np.isnan(expert_preds)), "expert_pred contains NaN"

    def test_is_mixture_and_num_experts(self, trained_mixture_model):
        """Test is_mixture() and num_experts() methods."""
        bst, _ = trained_mixture_model

        assert bst.is_mixture() is True
        assert bst.num_experts() == 3


class TestMixtureSaveLoad:
    """Test MoE model save/load functionality."""

    def test_save_load_predict_consistency(self, tmp_path):
        """save->load should produce consistent predict / predict_regime_proba."""
        X, y, _ = make_toy_regression_data(n_samples=150)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'boosting': 'mixture',
            'mixture_enable': True,
            'mixture_num_experts': 2,
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
        }

        # Train and save
        bst = lgb.train(params, train_data, num_boost_round=15)
        model_path = tmp_path / "mixture_model.txt"
        bst.save_model(str(model_path))

        # Get predictions before reload
        pred_before = bst.predict(X)
        regime_proba_before = bst.predict_regime_proba(X)
        regime_before = bst.predict_regime(X)
        expert_pred_before = bst.predict_expert_pred(X)

        # Load model
        bst_loaded = lgb.Booster(model_file=str(model_path))

        # Get predictions after reload
        pred_after = bst_loaded.predict(X)
        regime_proba_after = bst_loaded.predict_regime_proba(X)
        regime_after = bst_loaded.predict_regime(X)
        expert_pred_after = bst_loaded.predict_expert_pred(X)

        # Check consistency
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-6,
                                   err_msg="predict differs after save/load")
        np.testing.assert_allclose(regime_proba_before, regime_proba_after, rtol=1e-6,
                                   err_msg="predict_regime_proba differs after save/load")
        np.testing.assert_array_equal(regime_before, regime_after,
                                      err_msg="predict_regime differs after save/load")
        np.testing.assert_allclose(expert_pred_before, expert_pred_after, rtol=1e-6,
                                   err_msg="predict_expert_pred differs after save/load")

        # Check loaded model properties
        assert bst_loaded.is_mixture() is True
        assert bst_loaded.num_experts() == 2


class TestMixtureDisabled:
    """Test that mixture_enable=0 matches standard GBDT behavior."""

    def test_mixture_disabled_matches_gbdt(self):
        """mixture_enable=0 should produce same results as standard GBDT."""
        X, y, _ = make_toy_regression_data(n_samples=100)
        train_data = lgb.Dataset(X, label=y)

        # Standard GBDT
        params_std = {
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
            'seed': 42,
        }
        bst_std = lgb.train(params_std, train_data, num_boost_round=10)
        pred_std = bst_std.predict(X)

        # GBDT with mixture_enable=0 (should behave same)
        params_moe_off = {
            'objective': 'regression',
            'mixture_enable': False,
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
            'seed': 42,
        }
        bst_moe_off = lgb.train(params_moe_off, train_data, num_boost_round=10)
        pred_moe_off = bst_moe_off.predict(X)

        # Predictions should be identical
        np.testing.assert_allclose(pred_std, pred_moe_off, rtol=1e-10,
                                   err_msg="mixture_enable=0 differs from standard GBDT")

        # Standard model should not be a mixture
        assert bst_std.is_mixture() is False
        assert bst_moe_off.is_mixture() is False


class TestMixtureNonMixtureErrors:
    """Test that MoE methods raise errors on non-mixture models."""

    def test_predict_regime_on_non_mixture_raises(self):
        """predict_regime on non-mixture model should raise error."""
        X, y, _ = make_toy_regression_data(n_samples=50)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
        }
        bst = lgb.train(params, train_data, num_boost_round=5)

        with pytest.raises(lgb.basic.LightGBMError):
            bst.predict_regime(X)

    def test_predict_regime_proba_on_non_mixture_raises(self):
        """predict_regime_proba on non-mixture model should raise error."""
        X, y, _ = make_toy_regression_data(n_samples=50)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'objective': 'regression',
            'verbose': -1,
            'num_leaves': 8,
        }
        bst = lgb.train(params, train_data, num_boost_round=5)

        with pytest.raises(lgb.basic.LightGBMError):
            bst.predict_regime_proba(X)


class TestMixtureWithDifferentObjectives:
    """Test MoE with different objective functions."""

    @pytest.mark.parametrize("objective", ["regression", "regression_l1", "huber"])
    def test_mixture_with_objective(self, objective):
        """Test MoE works with various regression objectives."""
        X, y, _ = make_toy_regression_data(n_samples=100)
        train_data = lgb.Dataset(X, label=y)

        params = {
            'boosting': 'mixture',
            'mixture_enable': True,
            'mixture_num_experts': 2,
            'objective': objective,
            'verbose': -1,
            'num_leaves': 8,
            'num_threads': 1,
        }

        bst = lgb.train(params, train_data, num_boost_round=10)
        pred = bst.predict(X)

        assert not np.any(np.isnan(pred))
        assert bst.is_mixture()
