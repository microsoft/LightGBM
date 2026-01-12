# Phase 2: C++ MixtureGBDT 実装

## 実施日
2026-01-11

## 作成ファイル

### 1. include/LightGBM/config.h
Mixture-of-Experts パラメータを追加:

```cpp
// Mixture-of-Experts Parameters
bool mixture_enable = false;           // MoEモードを有効化
int mixture_num_experts = 4;           // エキスパート数 (K)
double mixture_r_min = 1e-3;           // 責務の最小値（collapse防止）
int mixture_gate_iters_per_round = 1;  // ゲート学習のイテレーション数/ラウンド
std::string mixture_init = "uniform";  // 初期化方法

// E-step パラメータ
double mixture_e_step_alpha = 1.0;     // 尤度とpriorのバランス
std::string mixture_e_step_loss = "auto";  // E-stepの損失関数

// 時系列平滑化
std::string mixture_r_smoothing = "none";  // 平滑化方法
double mixture_smoothing_lambda = 0.0;         // EMA係数

// 出力モード
std::string mixture_predict_output = "value";

// Gate GBDT パラメータ
int mixture_gate_max_depth = 3;
int mixture_gate_num_leaves = 8;
double mixture_gate_learning_rate = 0.1;
double mixture_gate_lambda_l2 = 1.0;
```

### 2. src/boosting/mixture_gbdt.h
MixtureGBDT クラスのヘッダファイル:
- `GBDTBase` を継承
- K個のエキスパート GBDT (`experts_`)
- 1個のゲート GBDT (`gate_`)
- EM学習ループのメソッド (Forward, EStep, MStepExperts, MStepGate)
- 予測API (Predict, PredictRegime, PredictRegimeProba, PredictExpertPred)

### 3. src/boosting/mixture_gbdt.cpp
MixtureGBDT の実装:

#### 初期化 (Init)
- エキスパート config とゲート config を分離
- ゲートは multiclass (num_class = K)
- バッファ割り当て (responsibilities_, expert_pred_, gate_proba_, yhat_)

#### 学習ループ (TrainOneIter)
1. **Forward**: エキスパート予測とゲート確率を計算
2. **EStep**: 責務 r_ik を更新
   - s_ik = log(gate_proba_ik + eps) - alpha * loss(y_i, expert_pred_ik)
   - r_i = softmax(s_i)
   - r_min でクリップして再正規化
3. **SmoothResponsibilities**: EMA で時系列平滑化（オプション）
4. **MStepExperts**: 責務重み付き勾配でエキスパート学習
   - grad_k[i] = r_ik * grad[i]
   - hess_k[i] = r_ik * hess[i]
5. **MStepGate**: argmax(r) を擬似ラベルとしてゲート学習

#### 予測
- `Predict`: 重み付き和 yhat = Σ_k g_k * f_k
- `PredictRegime`: argmax gate probability
- `PredictRegimeProba`: gate probability (N × K)
- `PredictExpertPred`: expert predictions (N × K)

### 4. src/boosting/boosting.cpp
- `#include "mixture_gbdt.h"` を追加
- `CreateBoosting` に `type == "mixture"` を追加
- モデルファイルの `model_type == "mixture"` を処理

### 5. CMakeLists.txt
- `src/boosting/mixture_gbdt.cpp` を LGBM_SOURCES に追加

## 技術的な決定事項

### M-stepで r_ik を使う理由
仕様書通り、M-step Experts では g_k(x) ではなく r_ik を使用。
- g_k: ゲート出力（予測時のルーティング確率）
- r_ik: E-stepで計算された責務（「このサンプルをエキスパートkが担当すべき確率」）

レジームスイッチングモデルでは、各エキスパートが特定パターンを専門学習するため、
EMアルゴリズムの責務を使う設計が妥当。

### ゲート学習
- Hard label（argmax r_ik）を使用
- Multiclass cross-entropy の勾配を直接計算

## 未実装・TODO
- [ ] kmeans / residual_kmeans 初期化
- [ ] quantile loss の alpha 取得
- [ ] validation data の処理
- [ ] LoadModelFromString の完全実装
- [ ] DumpModel (JSON) の実装
- [ ] SHAP (PredictContrib) の実装

## 次のステップ
- ビルドテスト
- Phase 3: Python wrapper 実装
