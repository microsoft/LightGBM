#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive Benchmark: Regime Smoothing Approaches

Compares:
1. No smoothing (none)
2. EMA smoothing
3. Markov smoothing
4. Momentum smoothing (EMA with trend)

All with Optuna hyperparameter optimization.
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


def generate_synthetic_data(n_samples=2000, noise_level=0.5, seed=42):
    """Generate synthetic regime-switching data."""
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    y = np.zeros(n_samples)

    mask0 = regime_true == 0
    y[mask0] = (5.0 * X[mask0, 0] +
                3.0 * X[mask0, 0] * X[mask0, 2] +
                2.0 * np.sin(2 * X[mask0, 3]) + 10.0)

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


def evaluate_cv(X, y, params, n_splits=5, use_lagged=False, y_train_full=None):
    """Evaluate model with time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)

        try:
            model = lgb.train(params, train_data, num_boost_round=100)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
        except Exception as e:
            scores.append(float('inf'))

    return np.mean(scores)


def create_objective_none(X, y):
    """Objective for no smoothing."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'mixture',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
            'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
            'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 20),
            'mixture_r_smoothing': 'none',
        }
        return evaluate_cv(X, y, params)
    return objective


def create_objective_ema(X, y):
    """Objective for EMA smoothing."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'mixture',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
            'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
            'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 20),
            'mixture_r_smoothing': 'ema',
            'mixture_smoothing_lambda': trial.suggest_float('mixture_smoothing_lambda', 0.1, 0.9),
        }
        return evaluate_cv(X, y, params)
    return objective


def create_objective_markov(X, y):
    """Objective for Markov smoothing."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'mixture',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
            'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
            'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 20),
            'mixture_r_smoothing': 'markov',
            'mixture_smoothing_lambda': trial.suggest_float('mixture_smoothing_lambda', 0.1, 0.9),
        }
        return evaluate_cv(X, y, params)
    return objective


def create_objective_momentum(X, y):
    """Objective for momentum smoothing."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'boosting': 'mixture',
            'verbose': -1,
            'num_threads': 4,
            'seed': 42,
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
            'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
            'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 20),
            'mixture_r_smoothing': 'momentum',
            'mixture_smoothing_lambda': trial.suggest_float('mixture_smoothing_lambda', 0.1, 0.9),
        }
        return evaluate_cv(X, y, params)
    return objective


def run_benchmark(dataset_name, X, y, n_trials=30):
    """Run full benchmark for a dataset."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")

    methods = [
        ('None', create_objective_none),
        ('EMA', create_objective_ema),
        ('Markov', create_objective_markov),
        ('Momentum', create_objective_momentum),
    ]

    results = {}

    for i, (name, create_obj) in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] Optimizing {name} ({n_trials} trials)...")
        start_time = time.time()

        study = optuna.create_study(direction='minimize')
        study.optimize(create_obj(X, y), n_trials=n_trials, show_progress_bar=True)

        elapsed = time.time() - start_time
        results[name] = {
            'rmse': study.best_value,
            'params': study.best_params,
            'time': elapsed,
        }
        print(f"  Best RMSE: {study.best_value:.4f} ({elapsed:.1f}s)")

    return results


def main():
    print("="*70)
    print("Comprehensive Benchmark: All Regime Smoothing Approaches")
    print("="*70)

    n_trials = 30  # Reduced for faster execution
    all_results = {}

    # Dataset 1: Synthetic
    X, y, _ = generate_synthetic_data(n_samples=1500)
    all_results['Synthetic'] = run_benchmark('Synthetic (X→Regime)', X, y, n_trials)

    # Dataset 2: Hamilton GNP-like
    X, y, _ = generate_hamilton_gnp_data(n_samples=400)
    all_results['Hamilton'] = run_benchmark('Hamilton GNP-like', X, y, n_trials)

    # Summary
    print("\n" + "="*90)
    print("SUMMARY: Best RMSE by Method")
    print("="*90)

    methods = ['None', 'EMA', 'Markov', 'Momentum']
    header = f"{'Dataset':<15}" + "".join([f"{m:>12}" for m in methods])
    print(header)
    print("-"*70)

    for dataset, results in all_results.items():
        row = f"{dataset:<15}"
        best_rmse = min(r['rmse'] for r in results.values())
        for method in methods:
            rmse = results[method]['rmse']
            if rmse == best_rmse:
                row += f"{'**' + f'{rmse:.4f}' + '**':>12}"
            else:
                row += f"{rmse:>12.4f}"
        print(row)

    # Best method per dataset
    print("\n" + "="*70)
    print("Best Method per Dataset")
    print("="*70)
    for dataset, results in all_results.items():
        best_method = min(results, key=lambda m: results[m]['rmse'])
        best_rmse = results[best_method]['rmse']
        print(f"{dataset}: {best_method} (RMSE={best_rmse:.4f})")

    # Key parameters for best methods
    print("\n" + "="*70)
    print("Key Parameters for Best Methods")
    print("="*70)
    for dataset, results in all_results.items():
        best_method = min(results, key=lambda m: results[m]['rmse'])
        params = results[best_method]['params']
        print(f"\n{dataset} -> {best_method}:")
        for k, v in params.items():
            if 'mixture' in k or 'lambda' in k:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
