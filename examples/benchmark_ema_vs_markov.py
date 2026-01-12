#!/usr/bin/env python
# coding: utf-8
"""
Benchmark: EMA vs Markov Smoothing for MoE GBDT

This script compares EMA and Markov smoothing modes with full Optuna
hyperparameter optimization (50 trials each).

ベンチマーク: MoE GBDTのEMA vs Markovスムージング比較
Optunaによる完全なハイパーパラメータ最適化を実施
"""

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm_moe as lgb
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def generate_synthetic_data(n_samples=2000, noise_level=0.5, seed=42):
    """Generate synthetic regime-switching data where regime is determinable from X."""
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Regime determined by features (X → Regime)
    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    y = np.zeros(n_samples)

    # Regime 0: Positive relationship
    mask0 = regime_true == 0
    y[mask0] = (5.0 * X[mask0, 0] +
                3.0 * X[mask0, 0] * X[mask0, 2] +
                2.0 * np.sin(2 * X[mask0, 3]) + 10.0)

    # Regime 1: Negative relationship (fundamentally different)
    mask1 = regime_true == 1
    y[mask1] = (-5.0 * X[mask1, 0] -
                2.0 * X[mask1, 1]**2 +
                3.0 * np.cos(2 * X[mask1, 4]) - 10.0)

    y += np.random.randn(n_samples) * noise_level

    return X, y, regime_true


def generate_hamilton_gnp_data(n_samples=500, seed=42):
    """Generate Hamilton GNP-like regime-switching data."""
    np.random.seed(seed)
    n_features = 4
    X = np.random.randn(n_samples, n_features)

    # Time-based regime switching (latent)
    t = np.arange(n_samples)
    regime_prob = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    # Expansion regime
    mask0 = regime_true == 0
    y[mask0] = 0.8 + 0.3 * X[mask0, 0] + 0.2 * X[mask0, 1]

    # Recession regime
    mask1 = regime_true == 1
    y[mask1] = -0.5 + 0.1 * X[mask1, 0] - 0.3 * X[mask1, 2]

    y += np.random.randn(n_samples) * 0.3

    return X, y, regime_true


def generate_vix_data(n_samples=1000, seed=42):
    """Generate VIX-like volatility regime data."""
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Volatility regime (high/low)
    t = np.arange(n_samples)
    regime_prob = 0.3 + 0.4 * (np.sin(2 * np.pi * t / 200) > 0)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    # Low volatility regime
    mask0 = regime_true == 0
    y[mask0] = 0.01 + 0.002 * np.abs(X[mask0, 0])

    # High volatility regime
    mask1 = regime_true == 1
    y[mask1] = 0.025 + 0.005 * np.abs(X[mask1, 0]) + 0.003 * X[mask1, 1]**2

    y += np.random.randn(n_samples) * 0.005

    return X, y, regime_true


def evaluate_cv(X, y, params, n_splits=5, use_markov_predict=False):
    """Evaluate model with time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)

        try:
            model = lgb.train(params, train_data, num_boost_round=100)

            if use_markov_predict and params.get('mixture_r_smoothing') == 'markov':
                pred = model.predict_markov(X_val)
            else:
                pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
        except Exception as e:
            scores.append(float('inf'))

    return np.mean(scores)


def create_objective(X, y, smoothing_mode):
    """Create Optuna objective function for given smoothing mode."""

    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'mixture',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,

            # Standard GBDT hyperparameters
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),

            # MoE hyperparameters
            'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
            'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
            'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),

            # Smoothing parameters
            'mixture_r_smoothing': smoothing_mode,
        }

        # Add lambda for EMA/Markov smoothing
        if smoothing_mode in ['ema', 'markov']:
            params['mixture_smoothing_lambda'] = trial.suggest_float('mixture_smoothing_lambda', 0.1, 0.9)

        use_markov = (smoothing_mode == 'markov')
        return evaluate_cv(X, y, params, use_markov_predict=use_markov)

    return objective


def create_std_objective(X, y):
    """Create Optuna objective for standard GBDT."""

    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'gbdt',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,

            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

        return evaluate_cv(X, y, params)

    return objective


def run_benchmark(dataset_name, X, y, n_trials=50):
    """Run full benchmark for a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")

    results = {}

    # Standard GBDT
    print(f"\n[1/4] Optimizing Standard GBDT ({n_trials} trials)...")
    study_std = optuna.create_study(direction='minimize')
    study_std.optimize(create_std_objective(X, y), n_trials=n_trials, show_progress_bar=True)
    results['Standard GBDT'] = {
        'rmse': study_std.best_value,
        'params': study_std.best_params
    }
    print(f"  Best RMSE: {study_std.best_value:.4f}")

    # MoE (none)
    print(f"\n[2/4] Optimizing MoE (no smoothing) ({n_trials} trials)...")
    study_none = optuna.create_study(direction='minimize')
    study_none.optimize(create_objective(X, y, 'none'), n_trials=n_trials, show_progress_bar=True)
    results['MoE (none)'] = {
        'rmse': study_none.best_value,
        'params': study_none.best_params
    }
    print(f"  Best RMSE: {study_none.best_value:.4f}")

    # MoE (EMA)
    print(f"\n[3/4] Optimizing MoE (EMA) ({n_trials} trials)...")
    study_ema = optuna.create_study(direction='minimize')
    study_ema.optimize(create_objective(X, y, 'ema'), n_trials=n_trials, show_progress_bar=True)
    results['MoE (EMA)'] = {
        'rmse': study_ema.best_value,
        'params': study_ema.best_params
    }
    print(f"  Best RMSE: {study_ema.best_value:.4f}")
    if 'mixture_smoothing_lambda' in study_ema.best_params:
        print(f"  Best lambda: {study_ema.best_params['mixture_smoothing_lambda']:.3f}")

    # MoE (Markov)
    print(f"\n[4/4] Optimizing MoE (Markov) ({n_trials} trials)...")
    study_markov = optuna.create_study(direction='minimize')
    study_markov.optimize(create_objective(X, y, 'markov'), n_trials=n_trials, show_progress_bar=True)
    results['MoE (Markov)'] = {
        'rmse': study_markov.best_value,
        'params': study_markov.best_params
    }
    print(f"  Best RMSE: {study_markov.best_value:.4f}")
    if 'mixture_smoothing_lambda' in study_markov.best_params:
        print(f"  Best lambda: {study_markov.best_params['mixture_smoothing_lambda']:.3f}")

    return results


def main():
    print("="*60)
    print("EMA vs Markov Smoothing Benchmark")
    print("Full Optuna Hyperparameter Optimization (50 trials)")
    print("="*60)

    n_trials = 50
    all_results = {}

    # Dataset 1: Synthetic (X → Regime)
    X, y, _ = generate_synthetic_data(n_samples=2000)
    all_results['Synthetic (X→Regime)'] = run_benchmark('Synthetic (X→Regime)', X, y, n_trials)

    # Dataset 2: Hamilton GNP-like
    X, y, _ = generate_hamilton_gnp_data(n_samples=500)
    all_results['Hamilton GNP-like'] = run_benchmark('Hamilton GNP-like', X, y, n_trials)

    # Dataset 3: VIX-like
    X, y, _ = generate_vix_data(n_samples=1000)
    all_results['VIX Volatility'] = run_benchmark('VIX Volatility', X, y, n_trials)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Best RMSE (with Optuna optimization)")
    print("="*80)
    print(f"{'Dataset':<25} {'Std GBDT':>12} {'MoE(none)':>12} {'MoE(EMA)':>12} {'MoE(Markov)':>12}")
    print("-"*80)

    for dataset, results in all_results.items():
        std = results['Standard GBDT']['rmse']
        none = results['MoE (none)']['rmse']
        ema = results['MoE (EMA)']['rmse']
        markov = results['MoE (Markov)']['rmse']

        # Find best
        best_val = min(std, none, ema, markov)
        std_str = f"**{std:.4f}**" if std == best_val else f"{std:.4f}"
        none_str = f"**{none:.4f}**" if none == best_val else f"{none:.4f}"
        ema_str = f"**{ema:.4f}**" if ema == best_val else f"{ema:.4f}"
        markov_str = f"**{markov:.4f}**" if markov == best_val else f"{markov:.4f}"

        print(f"{dataset:<25} {std_str:>12} {none_str:>12} {ema_str:>12} {markov_str:>12}")

    # EMA vs Markov comparison
    print("\n" + "="*80)
    print("EMA vs Markov Comparison")
    print("="*80)
    for dataset, results in all_results.items():
        ema_rmse = results['MoE (EMA)']['rmse']
        markov_rmse = results['MoE (Markov)']['rmse']
        diff_pct = (ema_rmse - markov_rmse) / ema_rmse * 100

        ema_lambda = results['MoE (EMA)']['params'].get('mixture_smoothing_lambda', None)
        markov_lambda = results['MoE (Markov)']['params'].get('mixture_smoothing_lambda', None)

        ema_lambda_str = f"{ema_lambda:.3f}" if ema_lambda is not None else "N/A"
        markov_lambda_str = f"{markov_lambda:.3f}" if markov_lambda is not None else "N/A"

        winner = "Markov" if markov_rmse < ema_rmse else "EMA"
        print(f"\n{dataset}:")
        print(f"  EMA:    RMSE={ema_rmse:.4f}, lambda={ema_lambda_str}")
        print(f"  Markov: RMSE={markov_rmse:.4f}, lambda={markov_lambda_str}")
        print(f"  Winner: {winner} ({abs(diff_pct):.1f}% {'better' if diff_pct > 0 else 'worse'})")


if __name__ == '__main__':
    main()
