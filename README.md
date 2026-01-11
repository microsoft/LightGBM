<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=300 />

LightGBM-MoE
============

**A regime-switching / Mixture-of-Experts extension of LightGBM.**

[English](#english) | [Japanese (日本語)](#japanese)

---

<a name="english"></a>
## English

### Overview

LightGBM-MoE is a fork of [Microsoft LightGBM](https://github.com/microsoft/LightGBM) that implements **Mixture-of-Experts (MoE) / Regime-Switching GBDT** natively in C++.

Unlike standard GBDT which fits a single model to all data, MoE learns **K specialized expert models** plus a **gate model** that dynamically routes each sample to the appropriate expert(s).

### Mathematical Formulation

#### Standard GBDT
```
ŷ(x) = f(x)
```
A single GBDT model `f` predicts for all inputs.

#### MoE-GBDT (This Implementation)
```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

Where:
- `fₖ(x)`: Expert k's prediction (K regression GBDTs)
- `gₖ(x)`: Gate's routing probability for expert k (softmax output from multiclass GBDT)
- `K`: Number of experts (hyperparameter)

The final prediction is a **soft mixture** of expert predictions weighted by the gate's routing probabilities.

### Training: EM-style Algorithm

Each boosting iteration performs:

1. **Forward Pass**: Compute expert predictions and gate probabilities
2. **E-step**: Update responsibility `r_ik` (how much expert k is responsible for sample i)
   ```
   s_ik = log(g_ik + ε) - α · L(yᵢ, f_k(xᵢ))
   r_ik = softmax(s_i)
   ```
3. **M-step Experts**: Train each expert with weighted gradients
   ```
   grad_k[i] = r_ik · grad[i]
   hess_k[i] = r_ik · hess[i]
   ```
4. **M-step Gate**: Update gate using pseudo-labels `z_i = argmax_k(r_ik)`

### Installation

```bash
# Clone
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE

# Build
mkdir build && cd build
cmake ..
make -j4

# Install Python package
cd ../python-package
pip install -e .
```

### Usage

```python
import lightgbm_moe as lgb

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)

# MoE parameters
params = {
    'boosting': 'mixture',           # Enable MoE mode
    'mixture_num_experts': 4,        # Number of experts (K)
    'objective': 'regression',
    'num_leaves': 31,
}

# Train
model = lgb.train(params, train_data, num_boost_round=100)

# Predictions
y_pred = model.predict(X_test)                    # Weighted mixture prediction
regime_proba = model.predict_regime_proba(X_test) # Gate probabilities (N x K)
regime = model.predict_regime(X_test)             # Most likely regime (argmax)
expert_preds = model.predict_expert_pred(X_test)  # Individual expert predictions (N x K)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boosting` | string | `"gbdt"` | Set to `"mixture"` for MoE mode |
| `mixture_num_experts` | int | 4 | Number of expert models (K) |
| `mixture_e_step_alpha` | float | 1.0 | Weight for loss term in E-step |
| `mixture_r_min` | float | 1e-3 | Minimum responsibility (prevents collapse) |
| `mixture_r_smoothing` | string | `"none"` | `"none"` or `"ema"` for temporal smoothing |
| `mixture_r_ema_lambda` | float | 0.0 | EMA coefficient (0-1) for responsibility smoothing |

### New Prediction APIs

| Method | Output Shape | Description |
|--------|--------------|-------------|
| `predict(X)` | (N,) | Standard prediction (weighted mixture) |
| `predict_regime_proba(X)` | (N, K) | Gate probabilities for each expert |
| `predict_regime(X)` | (N,) | Most likely regime index |
| `predict_expert_pred(X)` | (N, K) | Each expert's individual prediction |

### Assumptions

1. **Regression only**: Currently supports regression objectives (MSE, MAE, Huber, etc.)
2. **Row order for EMA**: If using `mixture_r_smoothing="ema"`, data must be sorted by time
3. **Single time series**: EMA smoothing assumes single series (no cross-sectional mixing)

### Performance Benchmark

**Environment**: 10,000 samples, 20 features, 50 boosting rounds, 4 threads

| Model | Training Time | Predict Time | Overhead |
|-------|--------------|--------------|----------|
| Standard GBDT | 0.10s | 3.97ms | 1.0x |
| MoE (K=4) | 0.44s | 18.69ms | ~4.5x |

The overhead is **near-optimal**: with K=4 experts + 1 gate = 5 models, ~4-5x overhead is expected. The implementation uses OpenMP parallelization in 7 critical sections.

### Regime-Switching Benchmark

**When to use MoE**: MoE excels when data has **clear, distinct regime structure** where different regimes follow fundamentally different functions.

#### Synthetic Data (Clear Regime Structure)

**Setup**: 2,000 samples with 2 regimes (Bull/Bear), 5 features, 150 boosting rounds

| Model | RMSE | R² | Improvement |
|-------|------|-----|-------------|
| Standard GBDT | 5.19 | 0.821 | - |
| MoE (K=2, α=1.0) | 4.49 | 0.867 | **+13.6%** |

![Regime Switching Comparison](examples/regime_switching_comparison.png)

Run the demo: `python examples/regime_switching_demo.py`

#### Real Financial Data (Weak Regime Structure)

| Task | Standard GBDT | MoE (K=2) | Result |
|------|---------------|-----------|--------|
| Daily Return Prediction (6 tickers) | baseline | -1.6% | Standard wins |
| 5-day Return (S&P 500 + VIX) | baseline | -2.9% | Standard wins |
| Volatility Prediction | R²=0.333 | R²=0.312 | Standard wins |

**Key insight**: MoE is **not universally better**. It excels when:
- Data has clear, separable regime structure
- Different regimes follow fundamentally different functions
- Regimes can be learned from features (not purely latent)

For general prediction tasks without clear regime structure, standard GBDT may perform better due to lower model complexity.

---

<a name="japanese"></a>
## Japanese (日本語)

### 概要

LightGBM-MoE は [Microsoft LightGBM](https://github.com/microsoft/LightGBM) のフォークで、**Mixture-of-Experts (MoE) / レジームスイッチング GBDT** をC++でネイティブ実装しています。

標準GBDTが全データに対して単一モデルを学習するのに対し、MoEは**K個の専門家モデル**と、各サンプルを適切な専門家にルーティングする**ゲートモデル**を学習します。

### 数式

#### 標準GBDT
```
ŷ(x) = f(x)
```
単一のGBDTモデル `f` が全ての入力に対して予測。

#### MoE-GBDT（本実装）
```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

各記号の意味:
- `fₖ(x)`: エキスパートkの予測値（K個の回帰GBDT）
- `gₖ(x)`: ゲートのルーティング確率（多クラスGBDTのsoftmax出力）
- `K`: エキスパート数（ハイパーパラメータ）

最終予測は、ゲートのルーティング確率で重み付けされたエキスパート予測の**ソフトミクスチャ**です。

### 学習: EM風アルゴリズム

各ブースティングイテレーションで以下を実行:

1. **Forward Pass**: エキスパート予測とゲート確率を計算
2. **E-step**: 責務 `r_ik`（サンプルiに対するエキスパートkの責任度）を更新
   ```
   s_ik = log(g_ik + ε) - α · L(yᵢ, f_k(xᵢ))
   r_ik = softmax(s_i)
   ```
3. **M-step Experts**: 重み付き勾配で各エキスパートを学習
   ```
   grad_k[i] = r_ik · grad[i]
   hess_k[i] = r_ik · hess[i]
   ```
4. **M-step Gate**: 擬似ラベル `z_i = argmax_k(r_ik)` でゲートを更新

### インストール

```bash
# クローン
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE

# ビルド
mkdir build && cd build
cmake ..
make -j4

# Pythonパッケージインストール
cd ../python-package
pip install -e .
```

### 使用方法

```python
import lightgbm_moe as lgb

# データセット作成
train_data = lgb.Dataset(X_train, label=y_train)

# MoEパラメータ
params = {
    'boosting': 'mixture',           # MoEモード有効化
    'mixture_num_experts': 4,        # エキスパート数（K）
    'objective': 'regression',
    'num_leaves': 31,
}

# 学習
model = lgb.train(params, train_data, num_boost_round=100)

# 予測
y_pred = model.predict(X_test)                    # 重み付きミクスチャ予測
regime_proba = model.predict_regime_proba(X_test) # ゲート確率 (N x K)
regime = model.predict_regime(X_test)             # 最も可能性の高いレジーム (argmax)
expert_preds = model.predict_expert_pred(X_test)  # 各エキスパートの予測 (N x K)
```

### 主要パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `boosting` | string | `"gbdt"` | MoEモードには `"mixture"` を指定 |
| `mixture_num_experts` | int | 4 | エキスパートモデル数（K） |
| `mixture_e_step_alpha` | float | 1.0 | E-stepでの損失項の重み |
| `mixture_r_min` | float | 1e-3 | 最小責務（崩壊防止） |
| `mixture_r_smoothing` | string | `"none"` | `"none"` または時系列平滑化用 `"ema"` |
| `mixture_r_ema_lambda` | float | 0.0 | 責務平滑化のEMA係数（0-1） |

### 新しい予測API

| メソッド | 出力形状 | 説明 |
|--------|--------------|-------------|
| `predict(X)` | (N,) | 標準予測（重み付きミクスチャ） |
| `predict_regime_proba(X)` | (N, K) | 各エキスパートへのゲート確率 |
| `predict_regime(X)` | (N,) | 最も可能性の高いレジームインデックス |
| `predict_expert_pred(X)` | (N, K) | 各エキスパートの個別予測 |

### 前提条件

1. **回帰のみ**: 現在は回帰目的関数のみサポート（MSE, MAE, Huber等）
2. **EMA用の行順序**: `mixture_r_smoothing="ema"` 使用時はデータが時系列順である必要あり
3. **単一時系列**: EMA平滑化は単一系列を想定（クロスセクション混合なし）

### パフォーマンスベンチマーク

**環境**: 10,000サンプル, 20特徴量, 50ブースティングラウンド, 4スレッド

| モデル | 学習時間 | 予測時間 | オーバーヘッド |
|-------|---------|---------|--------------|
| 標準GBDT | 0.10秒 | 3.97ms | 1.0x |
| MoE (K=4) | 0.44秒 | 18.69ms | 約4.5x |

オーバーヘッドは**ほぼ理論最適値**: K=4エキスパート + 1ゲート = 5モデルで、約4-5倍は想定通り。実装では7箇所の重要セクションでOpenMP並列化を使用。

### レジームスイッチング・ベンチマーク

**MoEを使うべき場面**: MoEは、異なるレジームが根本的に異なる関数に従う**明確なレジーム構造**を持つデータで優れた性能を発揮します。

#### 合成データ（明確なレジーム構造あり）

**設定**: 2,000サンプル、2レジーム（強気/弱気）、5特徴量、150ブースティングラウンド

| モデル | RMSE | R² | 改善率 |
|-------|------|-----|--------|
| 標準GBDT | 5.19 | 0.821 | - |
| MoE (K=2, α=1.0) | 4.49 | 0.867 | **+13.6%** |

![レジームスイッチング比較](examples/regime_switching_comparison.png)

デモ実行: `python examples/regime_switching_demo.py`

#### 実金融データ（弱いレジーム構造）

| タスク | 標準GBDT | MoE (K=2) | 結果 |
|--------|----------|-----------|------|
| 日次リターン予測（6銘柄） | 基準 | -1.6% | 標準が勝利 |
| 5日リターン（S&P 500 + VIX） | 基準 | -2.9% | 標準が勝利 |
| ボラティリティ予測 | R²=0.333 | R²=0.312 | 標準が勝利 |

**重要な知見**: MoEは**万能ではありません**。以下の条件で有効：
- データに明確で分離可能なレジーム構造がある
- 異なるレジームが根本的に異なる関数に従う
- レジームが特徴量から学習可能（純粋に潜在的ではない）

明確なレジーム構造のない一般的な予測タスクでは、モデルの複雑さが低い標準GBDTの方が良い性能を発揮する可能性があります。

---

## Original LightGBM

This project is a fork of [LightGBM](https://github.com/microsoft/LightGBM) by Microsoft.

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy
- Support of parallel, distributed, and GPU learning

For the original LightGBM documentation, please refer to https://lightgbm.readthedocs.io/

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for additional details.
