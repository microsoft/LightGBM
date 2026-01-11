# Phase 1: Python パッケージリブランド

## 実施日
2026-01-11

## 変更内容

### 1. ディレクトリリネーム
```bash
cd python-package
git mv lightgbm lightgbm_moe
```

### 2. pyproject.toml の変更
- `name`: `lightgbm` → `lightgbm-moe`
- `description`: LightGBM-MoE の説明に更新
- `tool.ruff.lint.isort.known-first-party`: `lightgbm` → `lightgbm_moe`
- `tool.ruff.lint.per-file-ignores`: パス更新

### 3. __init__.py の変更
- docstring を LightGBM-MoE の説明に更新

### 4. compat.py の変更
- コメント内の `import lightgbm` 参照を `import lightgbm_moe` に更新

### 5. README.md の変更
- LightGBM-MoE のタイトルと説明を追加

## 変更後のインポート方法
```python
import lightgbm_moe
# または
from lightgbm_moe import LGBMRegressor
```

## 変更後のインストール方法
```bash
pip install lightgbm-moe
# または開発モード
pip install -e python-package/
```

## 確認事項（ビルド後）
- [ ] `import lightgbm_moe` が通る
- [ ] `pip show lightgbm-moe` で正しいパッケージ名が表示される
- [ ] 基本的な train/predict が動作する

## 備考
- C++共有ライブラリ（lib_lightgbm）の名前は変更しない（仕様通り）
- 内部namespace/C APIも変更しない（仕様通り）
