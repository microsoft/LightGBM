# Phase 3: Python Wrapper 実装

## 実施日
2026-01-11

## 変更ファイル

### 1. include/LightGBM/c_api.h

#### 新規定数
```c
#define C_API_PREDICT_REGIME        (4)  // MoE用: regime argmax
#define C_API_PREDICT_REGIME_PROBA  (5)  // MoE用: regime probabilities
#define C_API_PREDICT_EXPERT_PRED   (6)  // MoE用: expert predictions
```

#### 新規API関数
```c
// MoEモデルかどうかをチェック
int LGBM_BoosterIsMixture(BoosterHandle handle, int* out_is_mixture);

// エキスパート数を取得
int LGBM_BoosterGetNumExperts(BoosterHandle handle, int* out_num_experts);

// レジーム予測 (argmax)
int LGBM_BoosterPredictRegime(BoosterHandle handle, ...);

// レジーム確率予測
int LGBM_BoosterPredictRegimeProba(BoosterHandle handle, ...);

// エキスパート個別予測
int LGBM_BoosterPredictExpertPred(BoosterHandle handle, ...);
```

### 2. src/c_api.cpp

`mixture_gbdt.h` を include し、上記APIの実装を追加:
- `dynamic_cast<MixtureGBDT*>` でMoEモデルかどうかを判定
- 行ごとにデータを取得し、MixtureGBDTの予測メソッドを呼び出す
- OpenMP並列化

### 3. python-package/lightgbm_moe/basic.py

Boosterクラスに新規メソッドを追加:

```python
def is_mixture(self) -> bool:
    """MoEモデルかどうかをチェック"""

def num_experts(self) -> int:
    """エキスパート数を取得"""

def predict_regime(self, data, **kwargs) -> np.ndarray:
    """レジーム予測 (argmax)
    Returns: shape (n_samples,), dtype=int32
    """

def predict_regime_proba(self, data, **kwargs) -> np.ndarray:
    """レジーム確率予測
    Returns: shape (n_samples, n_experts), dtype=float64
    """

def predict_expert_pred(self, data, **kwargs) -> np.ndarray:
    """エキスパート個別予測
    Returns: shape (n_samples, n_experts), dtype=float64
    """
```

## 使用例

```python
import lightgbm_moe as lgbm

# MoEモデルの学習
params = {
    'boosting': 'mixture',
    'mixture_enable': True,
    'mixture_num_experts': 4,
    'objective': 'regression',
}
train_data = lgbm.Dataset(X_train, y_train)
model = lgbm.train(params, train_data, num_boost_round=100)

# 通常の予測
y_pred = model.predict(X_test)

# レジーム予測
regimes = model.predict_regime(X_test)  # shape (n_samples,)

# レジーム確率
regime_proba = model.predict_regime_proba(X_test)  # shape (n_samples, 4)

# 各エキスパートの予測
expert_preds = model.predict_expert_pred(X_test)  # shape (n_samples, 4)
```

## 備考

- `predict_regime`, `predict_regime_proba`, `predict_expert_pred` はMoEモデルでのみ使用可能
- 通常のGBDTモデルで呼び出すと `LightGBMError` が発生
- データ型はfloat64に変換される

## 次のステップ

- Phase 4: モデル保存/読み込み、テスト
