目的:
LightGBMをforkし、Mixture-of-GBDT（K experts + 1 gate）を“内部実装”する（外部ラッパー禁止）。目的変数は回帰で、MSE固定ではなく LightGBM の objective を踏襲する。predictは従来のyに加え、レジーム推定（regime argmax と regime_proba）も返せるようにする。
加えて、フォーク後の命名は「レポジトリ名変更 + Pythonの import 名を xxx に変更」だけを行う（C++共有ライブラリ名/C API/内部namespace/モデル形式は当面LightGBMのままでよい）。PyPI衝突回避のため、wheel名/配布名（pip install名）も xxx に変更する。
また、E-stepのalpha、rの時系列平滑化（EMA）、専門家本数Kはすべてハイパラとして外部から調整可能にする。
時系列は「1銘柄（単一系列）」に限定し、timestampはPython側でpd.Timestampとして保持するが、LightGBM内部には渡さない。学習・予測に渡す行順がtimestamp昇順であることを前提に、r平滑化は行順でEMA適用する。

──────────────────────────────────────────────────────────────────────────────
0) リブランド要件（最初にやる / 変更範囲はPythonのみ）
- リブランドは, LightGBM-MoEとする（以下, xxxと記載しているが）
- GitHub repo名は新名称に変更（コード変更不要だがREADME等の表記更新はする）
- Python:
  - import は `import xxx` にする（`python-package/lightgbm/` を `python-package/xxx/` に改名）
  - wheel / distribution name（pip installの名前）も `xxx` にする（PyPIで lightgbm と衝突しない）
  - 共有ライブラリのロード（lib_lightgbm を読む等）は当面そのままでよい
  - 互換の `import lightgbm` stub は不要（作らない）
- Readmeに , こんな感じで書き足す
    - Lightgbm-moe: A regime-switching / mixture-of-experts extension of LightGBM. 
0.1) Python リネーム実装指示（具体）
- `python-package/lightgbm/` → `python-package/xxx/` に git mv
- `python-package` 配下のビルド設定を修正:
  - `pyproject.toml` があれば project.name / package 名を `xxx` に
  - もし `setup.py`/`setup.cfg` なら `name="xxx"` に変更、packages に `xxx` を含める
- `python-package/xxx/` 配下で `import lightgbm` / `from lightgbm` を `xxx` に置換
- ビルド後に `python -c "import xxx"` が通ることを確認
- `pip show xxx` ができ、dist名が xxx になっていることを確認

──────────────────────────────────────────────────────────────────────────────
1) モデル仕様（Mixture-of-GBDT）
- 予測: yhat(x) = Σ_k g_k(x) * f_k(x)
  - experts f_k: 回帰GBDTをK個
  - gate g_k: multiclass GBDT（softmax確率）
- 予測API:
  - 既存 predict は yhat のみ（互換維持）
  - 追加: predict_regime (argmax), predict_regime_proba (N×K)
  - 開発用: predict_expert_pred (N×K)（任意）

──────────────────────────────────────────────────────────────────────────────
2) 追加パラメータ（config/CLI/python共通）
Mixture基本:
- mixture_enable (bool, default false)
- mixture_num_experts (int K, default 4)              # ハイパラ
- mixture_r_min (double, default 1e-3)                # 責務下限クリップ
- mixture_gate_iters_per_round (int, default 1)
- mixture_init (string: uniform|kmeans|residual_kmeans, default uniform)

E-step（レジーム推定）:
- mixture_e_step_alpha (double, default 1.0)          # ハイパラ
- mixture_e_step_loss (string: l2|l1|quantile|auto, default auto)
  - auto: objective名から推定。推定できない場合はl2にフォールバック。

rの時系列平滑化（1銘柄・行順=時系列順 前提）:
- mixture_r_smoothing (string: none|ema, default none)
- mixture_smoothing_lambda (double, default 0.0)          # ハイパラ。0で平滑化なし

Predict出力:
- mixture_predict_output (string: value|value_and_regime|all, default value)
  - value: yhat
  - value_and_regime: yhat + argmax regime
  - all: yhat + regime_proba + expert_pred（開発用）

Gate専用パラメータ（prefixで衝突回避）:
- mixture_gate_max_depth, mixture_gate_num_leaves, mixture_gate_learning_rate, mixture_gate_lambda_l2, etc
  - 初期実装は安全な固定値でも可。TODOで外部指定できるようにする。

──────────────────────────────────────────────────────────────────────────────
3) loss の扱い（重要・確定 / MSE固定にしない）
- mixture全体のlossは LightGBM の ObjectiveFunction をそのまま使用する。
- 各 boosting iteration で:
  1) mixture 後の予測 yhat に対して objective_->GetGradients(yhat, y, grad, hess) を 1回だけ呼ぶ
  2) Expert k には grad_k = r_k * grad, hess_k = r_k * hess を渡して学習させる
- これにより regression_l2, l1, huber, quantile, poisson, tweedie, custom objective 等をそのまま使用可能
- Expertごとに異なるlossは設計外

──────────────────────────────────────────────────────────────────────────────
4) コード構造方針（最小改造）
- TreeLearner/Histogram/split探索などの低レベルは極力変更しない。
- 変更範囲は「Boostingの上位学習ループ」「Predictの出力」「モデルsave/load」「python wrapper」中心。
- 実装は新しい学習クラス MixtureGBDT（仮名）を追加し、通常GBDTと同様の外部I/Fで扱えるようにする。
- MixtureGBDT は内部に以下を持つ:
  - vector<unique_ptr<GBDT>> experts_ (K個)
  - unique_ptr<GBDT> gate_ (num_class=K の multiclass)
  - r_ (N×K), yhat_ (N), gate_proba_ (N×K), expert_pred_ (N×K)

──────────────────────────────────────────────────────────────────────────────
5) 学習ループ: EM風（E→Mを毎iter）
MixtureGBDT::TrainOneIter() の中身:

5.1 forward（予測の更新）
- expert_pred_[*][k] = experts_[k]->Predict(dataset)         # 可能ならバッチ
- gate_proba_[*][k] = gate_->PredictProba(dataset)           # softmax確率
- yhat = Σ_k gate_proba_k * expert_pred_k

5.2 E-step（責務 r 更新）
- スコア:
  s_ik = log(gate_proba_ik + eps) - alpha * loss_pointwise(y_i, expert_pred_ik)
  - alpha = mixture_e_step_alpha（ハイパラ）
  - eps = 1e-12
- loss_pointwise は mixture_e_step_loss / objective から決める:
  - l2: (y - pred)^2
  - l1: |y - pred|
  - quantile: pinball loss（tauは既存quantileパラメータから取得）
  - auto: objective名で推定、失敗ならl2にフォールバック
- r_i = softmax(s_i*)
  - 数値安定: s_i* から max を引いてから exp
- collapse防止:
  - r_ik = max(r_ik, mixture_r_min) して再正規化

5.3 rの時系列平滑化（オプション / 1銘柄で行順適用）
- mixture_r_smoothing == "ema" の場合:
  - lam = mixture_smoothing_lambda（ハイパラ）
  - 行順で適用（行順=timestamp昇順が外部で保証される前提）:
    for i=1..N-1:
      r[i] = (1-lam)*r[i] + lam*r[i-1]
    ※平滑化後に各行を再正規化（数値誤差対策）

5.4 M-step Experts（ObjectiveFunctionのgrad/hessを踏襲）
- objective_->GetGradients(yhat, y, grad, hess) を1回呼ぶ
- Expert k 用:
  grad_k[i] = r_ik * grad[i]
  hess_k[i] = r_ik * hess[i]
- experts_[k] を “外部からgrad/hessを注入して1iter学習できるAPI” で更新すること。
  例: GBDT::TrainOneIterCustom(grad_ptr, hess_ptr)
  - 既存の objective_function_->GetGradients() を呼ぶ経路を分岐し、内部の gradients_/hessians_ を差し替えられるようにする
  - TreeLearner/Histogramなどは流用し、split探索を壊さない

5.5 M-step Gate（擬似ラベルでmulticlass更新）
- z_i = argmax_k r_ik を擬似ラベルにして gate_ を学習
- gate_dataset の label を各iter更新する
  - metadata->SetLabel(z) 的な更新APIが無ければ追加する（頻繁更新なのでコピーコスト注意）
- gate_ を mixture_gate_iters_per_round 回 TrainOneIter()

5.6 metrics/early stopping
- mixture全体の yhat で train/valid のメトリクスを計算・表示（既存のeval機構に載せる）
- early stopping がある場合は mixtureのeval結果で動くように最小対応

──────────────────────────────────────────────────────────────────────────────
6) 予測APIの拡張（値 + レジーム）
互換性維持:
- 既存 Booster.predict は yhat のみを返す（戻り型変更しない）

追加API（C API + python wrapper）:
- predict_regime: argmax_k gate_proba（shape N, int）
- predict_regime_proba: gate_proba（shape N×K, float）
- 任意: predict_expert_pred（shape N×K）

CLI/C++出力モード（任意）:
- mixture_predict_output に応じて value / value_and_regime / all を出し分け可能にする
- pythonは別メソッドの方が安全（predictの戻り互換を守る）

──────────────────────────────────────────────────────────────────────────────
7) モデル保存/読み込み（1つのモデルとして保存）
- LightGBM model text に mixture セクションを追加:
  mixture_enable=1
  mixture_num_experts=K
  mixture_e_step_alpha=...
  mixture_e_step_loss=...
  mixture_r_smoothing=...
  mixture_smoothing_lambda=...
  [gate_model]
  ...（通常のLightGBMモデルdump）
  [expert_model_0]
  ...
  [expert_model_1]
  ...
- Load時に gate と experts を復元する
- mixture_enableが無いモデルは従来通り読む（後方互換）

──────────────────────────────────────────────────────────────────────────────
8) 受け入れテスト（必須）
- toyデータで mixture_enable=1, K=2 の学習が走り、NaNにならない
- predict_regime_proba の各行の和が1に近い（許容誤差1e-6）
- save→load後に predict / predict_regime_proba が一致（許容誤差）
- mixture_enable=0 のとき既存挙動と一致（回帰GBDTの予測同一）
- Python:
  - `import xxx` が通る
  - `pip show xxx` ができ、dist名が xxx になっている
  - 簡単なtrain/predictが動く

──────────────────────────────────────────────────────────────────────────────
9) 実装順序（重要）
1) python-package の import名/配布名変更（import xxx と pip show xxx 確認）
2) C++側で MixtureGBDT を実装して compile が通り toy で動く
3) python wrapper に predict_regime / predict_regime_proba を追加
4) save/load とテスト
5) 最後に性能最適化（バッチpredict、並列、メモリ削減）

補足（安定化の最低要件）
- r_minクリップ必須
- gateは浅く正則化強め（初期は固定でも可）
- softmax/log の数値安定化必須

grepの指針
- gradients_/hessians_ や objective_function_->GetGradients() をgrepして「grad/hessを作っている地点」を特定し、Mixture用の注入APIを追加する。
- SaveModelToString/LoadModelFromString をgrepして mixtureセクションを追加する。
- C API / python wrapper で predict を提供している箇所をgrepし、predict_regime系を追加する。
- python-package の build設定（pyproject/setup.py/setup.cfg）を編集し dist名を xxx に変える。