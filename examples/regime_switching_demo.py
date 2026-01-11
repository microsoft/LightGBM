#!/usr/bin/env python
# coding: utf-8
"""
Regime-Switching Demo: LightGBM vs LightGBM-MoE

This script demonstrates the advantage of MoE (Mixture-of-Experts) GBDT
over standard GBDT when the data has underlying regime structure.

レジームスイッチングデモ: LightGBM vs LightGBM-MoE
データに潜在的なレジーム構造がある場合のMoE GBDTの優位性を示します。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm_moe as lgb

# Set random seed for reproducibility
np.random.seed(42)


def generate_regime_switching_data(n_samples=3000, noise_level=0.5):
    """
    Generate synthetic data with complex regime structure.

    The key insight: standard GBDT learns a single function that compromises
    between regimes. MoE learns specialized experts for each regime.

    Regime 0 (Bull): y = 5*x0 + 3*x0*x2 + 2*sin(x3) + 10
                     (positive slope, interaction term, non-linearity)

    Regime 1 (Bear): y = -5*x0 - 2*x1^2 + 3*cos(x4) - 10
                     (negative slope, quadratic term, different non-linearity)

    Regime is determined by a combination of features (not trivially separable).
    """
    # Features
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Time index
    t = np.arange(n_samples)

    # Regime based on non-trivial combination of features
    # This makes it harder for standard GBDT to find the split
    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    # Generate y based on regime - fundamentally different functions
    y = np.zeros(n_samples)

    # Regime 0: Complex positive relationship
    mask0 = regime_true == 0
    y[mask0] = (5.0 * X[mask0, 0] +
                3.0 * X[mask0, 0] * X[mask0, 2] +  # Interaction term
                2.0 * np.sin(2 * X[mask0, 3]) +    # Non-linear term
                10.0)

    # Regime 1: Complex negative relationship (fundamentally different)
    mask1 = regime_true == 1
    y[mask1] = (-5.0 * X[mask1, 0] -
                2.0 * X[mask1, 1]**2 +              # Quadratic term
                3.0 * np.cos(2 * X[mask1, 4]) -     # Different non-linear
                10.0)

    # Add noise
    y += np.random.randn(n_samples) * noise_level

    return X, y, t, regime_true


def train_and_evaluate():
    """Train both models and create comparison visualizations."""

    # Generate data
    print("Generating regime-switching data...")
    X, y, t, regime_true = generate_regime_switching_data(n_samples=2000)

    # Split data (time-based split to simulate real forecasting)
    train_size = 1400
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    t_train, t_test = t[:train_size], t[train_size:]
    regime_train, regime_test = regime_true[:train_size], regime_true[train_size:]

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)

    # Common parameters
    common_params = {
        'objective': 'regression',
        'verbose': -1,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'num_threads': 4,
        'seed': 42,
    }

    # Train standard GBDT
    print("Training Standard GBDT...")
    params_std = {**common_params, 'boosting': 'gbdt'}
    model_std = lgb.train(params_std, train_data, num_boost_round=150)
    pred_std = model_std.predict(X_test)

    # Train MoE GBDT
    print("Training MoE GBDT (K=2)...")
    params_moe = {
        **common_params,
        'boosting': 'mixture',
        'mixture_num_experts': 2,
        'mixture_e_step_alpha': 1.0,  # Hard alpha (1.0) now works after gate indexing fix!
        'mixture_warmup_iters': 50,   # Warmup iterations for expert differentiation
    }
    # Train same number of rounds as standard GBDT
    model_moe = lgb.train(params_moe, train_data, num_boost_round=150)
    pred_moe = model_moe.predict(X_test)
    regime_pred = model_moe.predict_regime(X_test)
    regime_proba = model_moe.predict_regime_proba(X_test)

    # Calculate metrics
    metrics = {
        'Standard GBDT': {
            'RMSE': np.sqrt(mean_squared_error(y_test, pred_std)),
            'MAE': mean_absolute_error(y_test, pred_std),
            'R2': r2_score(y_test, pred_std),
        },
        'MoE GBDT (K=2)': {
            'RMSE': np.sqrt(mean_squared_error(y_test, pred_moe)),
            'MAE': mean_absolute_error(y_test, pred_moe),
            'R2': r2_score(y_test, pred_moe),
        }
    }

    print("\n" + "="*60)
    print("Performance Comparison / 性能比較")
    print("="*60)
    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {m['RMSE']:.4f}")
        print(f"  MAE:  {m['MAE']:.4f}")
        print(f"  R2:   {m['R2']:.4f}")

    improvement = (metrics['Standard GBDT']['RMSE'] - metrics['MoE GBDT (K=2)']['RMSE']) / metrics['Standard GBDT']['RMSE'] * 100
    print(f"\nRMSE Improvement: {improvement:.1f}%")

    # Key insight: MoE provides interpretability with comparable accuracy
    print("\n" + "="*60)
    print("Key Advantage: Interpretability / 主な利点：解釈可能性")
    print("="*60)
    print("MoE GBDT provides regime probabilities for each prediction,")
    print("enabling regime-aware analysis that standard GBDT cannot offer.")
    print("\nUse cases: Financial regime detection, market state analysis,")
    print("risk management with regime-conditional strategies.")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(
        t_test, y_test, pred_std, pred_moe,
        regime_test, regime_pred, regime_proba, metrics
    )

    return metrics


def create_visualizations(t_test, y_test, pred_std, pred_moe,
                         regime_test, regime_pred, regime_proba, metrics):
    """Create all comparison visualizations."""

    # Color maps for regimes (2 regimes now)
    num_regimes = 2
    regime_colors = ['#2ecc71', '#e74c3c']  # Green, Red
    regime_names = ['Regime 0 (Bull)', 'Regime 1 (Bear)']

    fig = plt.figure(figsize=(16, 14))

    # ============================================================
    # 1. Actual vs Predicted Scatter Plot
    # ============================================================
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.scatter(y_test, pred_std, alpha=0.5, label=f"Std GBDT (R²={metrics['Standard GBDT']['R2']:.3f})", s=20)
    ax1.scatter(y_test, pred_moe, alpha=0.5, label=f"MoE GBDT (R²={metrics['MoE GBDT (K=2)']['R2']:.3f})", s=20)

    # Perfect prediction line
    lims = [min(y_test.min(), pred_std.min(), pred_moe.min()),
            max(y_test.max(), pred_std.max(), pred_moe.max())]
    ax1.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')

    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('1. Actual vs Predicted / 実測値 vs 予測値')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # 2. Time Series: Standard GBDT
    # ============================================================
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t_test, y_test, 'k-', alpha=0.7, label='Actual', linewidth=1)
    ax2.plot(t_test, pred_std, 'b-', alpha=0.7, label='Predicted (Std)', linewidth=1)
    ax2.fill_between(t_test, y_test, pred_std, alpha=0.3, color='red')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title(f"2. Standard GBDT (RMSE={metrics['Standard GBDT']['RMSE']:.3f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # 3. Time Series: MoE GBDT
    # ============================================================
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(t_test, y_test, 'k-', alpha=0.7, label='Actual', linewidth=1)
    ax3.plot(t_test, pred_moe, 'g-', alpha=0.7, label='Predicted (MoE)', linewidth=1)
    ax3.fill_between(t_test, y_test, pred_moe, alpha=0.3, color='red')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.set_title(f"3. MoE GBDT (RMSE={metrics['MoE GBDT (K=2)']['RMSE']:.3f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ============================================================
    # 4. Regime Estimation Visualization (MoE only)
    # ============================================================
    ax4 = fig.add_subplot(3, 2, 4)

    # Plot actual values colored by true regime
    for r in range(num_regimes):
        mask = regime_test == r
        ax4.scatter(t_test[mask], y_test[mask], c=regime_colors[r],
                   alpha=0.6, s=15, label=f'Actual {regime_names[r]}')

    # Plot predictions colored by predicted regime (with different marker)
    for r in range(num_regimes):
        mask = regime_pred == r
        ax4.scatter(t_test[mask], pred_moe[mask], c=regime_colors[r],
                   alpha=0.6, s=15, marker='x')

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')
    ax4.set_title('4. Regime Estimation / レジーム推定\n(●: Actual, ×: Predicted)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # 5. Regime Probability Over Time
    # ============================================================
    ax5 = fig.add_subplot(3, 2, 5)

    for r in range(num_regimes):
        ax5.fill_between(t_test, 0, regime_proba[:, r], alpha=0.5,
                        color=regime_colors[r], label=regime_names[r])

    # Add true regime boundaries
    regime_changes = np.where(np.diff(regime_test) != 0)[0]
    for rc in regime_changes:
        ax5.axvline(x=t_test[rc], color='black', linestyle='--', alpha=0.5)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('Probability')
    ax5.set_title('5. Gate Probabilities Over Time / ゲート確率の時系列')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # ============================================================
    # 6. Prediction Error by Regime
    # ============================================================
    ax6 = fig.add_subplot(3, 2, 6)

    errors_std = []
    errors_moe = []

    for r in range(num_regimes):
        mask = regime_test == r
        errors_std.append(np.abs(y_test[mask] - pred_std[mask]))
        errors_moe.append(np.abs(y_test[mask] - pred_moe[mask]))

    x_pos = np.arange(num_regimes)
    width = 0.35

    ax6.bar(x_pos - width/2, [e.mean() for e in errors_std], width,
            label='Standard GBDT', color='steelblue', alpha=0.7)
    ax6.bar(x_pos + width/2, [e.mean() for e in errors_moe], width,
            label='MoE GBDT', color='forestgreen', alpha=0.7)

    ax6.set_xlabel('True Regime')
    ax6.set_ylabel('Mean Absolute Error')
    ax6.set_title('6. MAE by Regime / レジーム別MAE')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'Regime {i}' for i in range(num_regimes)])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('examples/regime_switching_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: examples/regime_switching_comparison.png")
    plt.close()

    # ============================================================
    # Additional: Confusion-like visualization for regime accuracy
    # ============================================================
    fig2, ax = plt.subplots(figsize=(8, 6))

    # Create confusion matrix
    from collections import Counter
    confusion = np.zeros((num_regimes, num_regimes))
    for true_r, pred_r in zip(regime_test, regime_pred):
        if true_r < num_regimes and pred_r < num_regimes:
            confusion[true_r, pred_r] += 1

    # Normalize
    confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)

    # Add text annotations
    for i in range(num_regimes):
        for j in range(num_regimes):
            text = ax.text(j, i, f'{confusion_norm[i, j]:.2f}\n({int(confusion[i, j])})',
                          ha='center', va='center', fontsize=12)

    ax.set_xticks(range(num_regimes))
    ax.set_yticks(range(num_regimes))
    ax.set_xticklabels([f'Pred {i}' for i in range(num_regimes)])
    ax.set_yticklabels([f'True {i}' for i in range(num_regimes)])
    ax.set_xlabel('Predicted Regime')
    ax.set_ylabel('True Regime')
    ax.set_title('Regime Classification Accuracy\nレジーム分類精度')

    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.savefig('examples/regime_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved: examples/regime_confusion_matrix.png")
    plt.close()


if __name__ == '__main__':
    metrics = train_and_evaluate()
    print("\nDone! Check the examples/ folder for visualizations.")
