# LightGBM-MoE 実装作業計画

## 概要

LightGBMをMixture-of-Experts（レジームスイッチングモデル）に拡張するプロジェクト。

**目的**: LightGBMをforkし、Mixture-of-GBDT（K experts + 1 gate）を内部実装する。

**予測式**: `yhat(x) = Σ_k g_k(x) * f_k(x)`
- experts `f_k`: 回帰GBDTをK個
- gate `g_k`: multiclass GBDT（softmax確率）

---

## 仕様書（version01.md）の技術的レビュー

### 検証済み項目（問題なし）

1. **モデル構造**: 標準的なMixture of Experts構造で適切
2. **E-stepの数式**: EM法における責任変数更新として妥当
   - `s_ik = log(gate_proba_ik + eps) - alpha * loss_pointwise(y_i, expert_pred_ik)`
   - `r_i = softmax(s_i*)`
   - alphaは尤度とpriorのバランス調整用ハイパーパラメータ

3. **loss_pointwiseの定義**: objective別に明確に定義されている
   - l2: `(y - pred)^2`
   - l1: `|y - pred|`
   - quantile: pinball loss

### 設計上の選択についての確認

#### M-step Expert更新で `r_ik` を使用する理由

仕様書では:
```
grad_k[i] = r_ik * grad[i]
hess_k[i] = r_ik * hess[i]
```

**考察**:
- 厳密な勾配降下なら `g_k(x_i)`（gate出力）を使うべき（chain rule）
- しかし `r_ik`（責任変数）を使うのはEM的アプローチとして妥当

**結論**: レジームスイッチングモデルの目的（各レジームが特定パターンを専門学習）を考えると、`r_ik`を使う設計は適切。各エキスパートは「自分が担当すべきサンプル」を重み付けして学習する。

---

## 実装フェーズ

### Phase 1: Pythonパッケージリブランド
- `python-package/lightgbm/` → `python-package/lightgbm_moe/`
- `pyproject.toml` の `name` を `lightgbm-moe` に変更
- import文を `import lightgbm_moe` に更新
- 確認: `pip install -e .` 後に `import lightgbm_moe` が動作

### Phase 2: C++ MixtureGBDT実装
- 新クラス `MixtureGBDT` を `src/boosting/` に追加
- 内部構造:
  - `vector<unique_ptr<GBDT>> experts_` (K個)
  - `unique_ptr<GBDT> gate_` (num_class=K)
  - `r_` (N×K), `yhat_` (N), `gate_proba_` (N×K), `expert_pred_` (N×K)
- EM学習ループ実装
- 設定パラメータ追加（`include/LightGBM/config.h`）

### Phase 3: Python wrapper拡張
- `predict_regime()`: argmax regime
- `predict_regime_proba()`: N×K確率行列
- C API拡張

### Phase 4: モデル保存/読み込み
- mixture セクションをモデルテキストに追加
- 後方互換性維持

### Phase 5: テスト・最適化
- 受け入れテスト実装
- パフォーマンス最適化

---

## ファイル構成

```
docs_claude/
├── 00_work_plan.md          # 本ファイル（作業計画）
├── 01_python_rebrand.md     # Phase 1 作業記録
├── 02_cpp_mixture_gbdt.md   # Phase 2 作業記録
├── 03_python_wrapper.md     # Phase 3 作業記録
├── 04_save_load_test.md     # Phase 4 作業記録
└── 05_optimization.md       # Phase 5 作業記録
```

---

## 進捗状況

| Phase | 内容 | 状態 |
|-------|------|------|
| 1 | Python パッケージリブランド | **完了** |
| 2 | C++ MixtureGBDT 実装 | **完了** |
| 3 | Python wrapper 実装 | **完了** |
| 4 | モデル保存/読み込み、テスト | 未着手 |
| 5 | パフォーマンス最適化 | 未着手 |

---

## 変更ファイル一覧

### Phase 1: リブランド
- `python-package/lightgbm/` → `python-package/lightgbm_moe/` (リネーム)
- `python-package/pyproject.toml` (パッケージ名変更)
- `README.md` (説明追加)

### Phase 2: C++ MixtureGBDT
- `include/LightGBM/config.h` (MoEパラメータ追加)
- `src/boosting/mixture_gbdt.h` (新規)
- `src/boosting/mixture_gbdt.cpp` (新規)
- `src/boosting/boosting.cpp` (MixtureGBDT登録)
- `CMakeLists.txt` (ソース追加)

### Phase 3: Python wrapper
- `include/LightGBM/c_api.h` (MoE API追加)
- `src/c_api.cpp` (MoE API実装)
- `python-package/lightgbm_moe/basic.py` (予測メソッド追加)

---

## 開始日時
2026-01-11
