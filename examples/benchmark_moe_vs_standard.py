#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive Benchmark: MoE vs Standard GBDT

Compares Standard GBDT vs MoE GBDT with full hyperparameter search.
Includes mixture_balance_factor in the MoE search space.

Usage:
    python examples/benchmark_moe_vs_standard.py
"""

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm_moe as lgb
import warnings
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 100
N_SPLITS = 5
NUM_BOOST_ROUND = 100


def generate_synthetic_data(n_samples=2000, noise_level=0.5, seed=42):
    """Generate synthetic regime-switching data where regime is determinable from X."""
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Regime is determined by features (X) - MoE should excel here
    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    y = np.zeros(n_samples)

    # Regime 0: positive, nonlinear
    mask0 = regime_true == 0
    y[mask0] = (5.0 * X[mask0, 0] +
                3.0 * X[mask0, 0] * X[mask0, 2] +
                2.0 * np.sin(2 * X[mask0, 3]) + 10.0)

    # Regime 1: negative, different nonlinearity
    mask1 = regime_true == 1
    y[mask1] = (-5.0 * X[mask1, 0] -
                2.0 * X[mask1, 1]**2 +
                3.0 * np.cos(2 * X[mask1, 4]) - 10.0)

    y += np.random.randn(n_samples) * noise_level

    return X, y, regime_true


def generate_hamilton_gnp_data(n_samples=500, seed=42):
    """Generate Hamilton GNP-like regime-switching data (latent regime)."""
    np.random.seed(seed)
    n_features = 4
    X = np.random.randn(n_samples, n_features)

    # Regime is latent (time-based probability, not from X)
    t = np.arange(n_samples)
    regime_prob = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    mask0 = regime_true == 0
    y[mask0] = 0.8 + 0.3 * X[mask0, 0] + 0.2 * X[mask0, 1]

    mask1 = regime_true == 1
    y[mask1] = -0.5 + 0.1 * X[mask1, 0] - 0.3 * X[mask1, 2]

    y += np.random.randn(n_samples) * 0.3

    return X, y, regime_true


def evaluate_cv(X, y, params, n_splits=N_SPLITS):
    """Evaluate model with time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)

        try:
            model = lgb.train(params, train_data, num_boost_round=NUM_BOOST_ROUND)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
        except Exception:
            scores.append(float('inf'))

    return np.mean(scores)


def create_objective_standard(X, y):
    """Objective for Standard GBDT with full hyperparameter search."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'gbdt',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,
            # Tree structure
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            # Learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            # Regularization
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            # Sampling
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
        }
        return evaluate_cv(X, y, params)
    return objective


def create_objective_moe(X, y):
    """Objective for MoE GBDT with full hyperparameter search including balance_factor."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'mixture',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,
            # Tree structure
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            # Learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            # Regularization
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            # Sampling
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
            # MoE specific
            'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
            'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
            'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
            'mixture_balance_factor': trial.suggest_int('mixture_balance_factor', 2, 10),
            # Smoothing
            'mixture_r_smoothing': trial.suggest_categorical('mixture_r_smoothing',
                                                             ['none', 'ema', 'markov', 'momentum']),
        }
        # Add smoothing lambda if smoothing is enabled
        if params['mixture_r_smoothing'] != 'none':
            params['mixture_smoothing_lambda'] = trial.suggest_float('mixture_smoothing_lambda', 0.1, 0.9)

        return evaluate_cv(X, y, params)
    return objective


def run_benchmark(dataset_name, X, y, regime_true, n_trials=N_TRIALS):
    """Run full benchmark comparing Standard vs MoE."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Regime distribution: {(regime_true == 0).mean():.1%} / {(regime_true == 1).mean():.1%}")

    results = {}

    # Standard GBDT
    print(f"\n[1/2] Optimizing Standard GBDT ({n_trials} trials)...")
    start_time = time.time()
    study_std = optuna.create_study(direction='minimize')
    study_std.optimize(create_objective_standard(X, y), n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - start_time
    results['Standard'] = {
        'rmse': study_std.best_value,
        'params': study_std.best_params,
        'time': elapsed,
    }
    print(f"  Best RMSE: {study_std.best_value:.4f} ({elapsed:.1f}s)")

    # MoE GBDT
    print(f"\n[2/2] Optimizing MoE GBDT ({n_trials} trials)...")
    start_time = time.time()
    study_moe = optuna.create_study(direction='minimize')
    study_moe.optimize(create_objective_moe(X, y), n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - start_time
    results['MoE'] = {
        'rmse': study_moe.best_value,
        'params': study_moe.best_params,
        'time': elapsed,
    }
    print(f"  Best RMSE: {study_moe.best_value:.4f} ({elapsed:.1f}s)")

    # Compare
    std_rmse = results['Standard']['rmse']
    moe_rmse = results['MoE']['rmse']
    improvement = (std_rmse - moe_rmse) / std_rmse * 100

    print(f"\n{'='*60}")
    print(f"Results for {dataset_name}:")
    print(f"{'='*60}")
    print(f"  Standard GBDT RMSE: {std_rmse:.4f}")
    print(f"  MoE GBDT RMSE:      {moe_rmse:.4f}")
    if improvement > 0:
        print(f"  MoE Improvement:    +{improvement:.1f}% ✓")
    else:
        print(f"  MoE Improvement:    {improvement:.1f}%")

    # Print MoE best params
    print(f"\nMoE Best Parameters:")
    moe_params = results['MoE']['params']
    print(f"  K (num_experts):    {moe_params.get('mixture_num_experts', 'N/A')}")
    print(f"  alpha:              {moe_params.get('mixture_e_step_alpha', 'N/A'):.2f}")
    print(f"  balance_factor:     {moe_params.get('mixture_balance_factor', 'N/A')}")
    print(f"  smoothing:          {moe_params.get('mixture_r_smoothing', 'N/A')}")
    if moe_params.get('mixture_r_smoothing', 'none') != 'none':
        print(f"  smoothing_lambda:   {moe_params.get('mixture_smoothing_lambda', 'N/A'):.2f}")

    return results


def main():
    print("="*80)
    print("Comprehensive Benchmark: MoE vs Standard GBDT")
    print(f"Trials: {N_TRIALS}, CV Splits: {N_SPLITS}, Boost Rounds: {NUM_BOOST_ROUND}")
    print("="*80)

    all_results = {}

    # Dataset 1: Synthetic (X → Regime)
    print("\n" + "="*80)
    print("Dataset 1: Synthetic (Regime determinable from X)")
    print("="*80)
    X, y, regime = generate_synthetic_data(n_samples=2000)
    all_results['Synthetic'] = run_benchmark('Synthetic (X→Regime)', X, y, regime)

    # Dataset 2: Hamilton GNP-like (latent regime)
    print("\n" + "="*80)
    print("Dataset 2: Hamilton GNP-like (Latent regime)")
    print("="*80)
    X, y, regime = generate_hamilton_gnp_data(n_samples=500)
    all_results['Hamilton'] = run_benchmark('Hamilton GNP-like', X, y, regime)

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<25} {'Std RMSE':>12} {'MoE RMSE':>12} {'Improvement':>12} {'Winner':>10}")
    print("-"*75)

    for dataset, results in all_results.items():
        std_rmse = results['Standard']['rmse']
        moe_rmse = results['MoE']['rmse']
        improvement = (std_rmse - moe_rmse) / std_rmse * 100
        winner = "MoE" if improvement > 0 else "Standard"

        print(f"{dataset:<25} {std_rmse:>12.4f} {moe_rmse:>12.4f} {improvement:>+11.1f}% {winner:>10}")

    print("\n" + "="*80)
    print("MoE Best Hyperparameters")
    print("="*80)
    for dataset, results in all_results.items():
        params = results['MoE']['params']
        print(f"\n{dataset}:")
        print(f"  K={params.get('mixture_num_experts')}, "
              f"alpha={params.get('mixture_e_step_alpha', 0):.2f}, "
              f"balance_factor={params.get('mixture_balance_factor')}, "
              f"smoothing={params.get('mixture_r_smoothing')}")


if __name__ == '__main__':
    main()
