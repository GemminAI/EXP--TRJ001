# FREEZE MANIFEST — EXP-2026-NVS-001 v1.3

本ファイルの全ハッシュ欄が埋まり、そのコミットに `v1.3-frozen` タグが付与された時点で凍結成立とする。
**凍結成立前に生成されたデータは本実験の結果として採用しない。**

## 凍結情報

| 項目 | 値 |
|---|---|
| 凍結日時 (UTC) | _未記入_ |
| 凍結コミット | _未記入_ |
| タグ | `v1.3-frozen` |
| 承認者 | _未記入_ |

## 環境

| 項目 | 値 |
|---|---|
| モデル | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| モデルコミットハッシュ | _未記入_ |
| vLLM バージョン | _未記入_ |
| CUDA / ドライバ | _未記入_ |
| GPU | _未記入_ |
| Python / 主要ライブラリ | _未記入_ |

## 成果物ハッシュ（SHA-256）

| # | 成果物 | パス | SHA-256 |
|---|---|---|---|
| 1 | 実験計画書 | `docs/EXP-2026-NVS-001_v1.3.md` | _未記入_ |
| 2 | 射影行列 W（4096×128） | `configs/W_pca128.npy` | _未記入_ |
| 3 | 中心化ベクトル μ（4096） | `configs/mu_pca.npy` | _未記入_ |
| 4 | キャリブレーションプロンプト 200 件 | `configs/calibration_prompts_v1.json` | _未記入_ |
| 5 | Type A 本実験プロンプト 1,000 件 | `configs/prompts_v1.json` | _未記入_ |
| 6 | 棄権・ヘッジ判定パターン | `configs/refusal_patterns_en.json` | _未記入_ |
| 7 | Type B 照合規則・正解集合 | `configs/typeB_reference.json` | _未記入_ |
| 8 | κ 計算モジュール | `src/compute_kappa.py` | _未記入_ |
| 9 | ラベル判定モジュール | `src/eval_hallucination.py` | _未記入_ |
| 10 | 生成設定 | `configs/generation_config.json` | _未記入_ |
| 11 | マスク手続き定義 | `configs/mask_protocol.json` | _未記入_ |
| 12 | Type B 用プロンプト 200 件（T1・v1.3 新設） | `configs/prompts_typeB_v1.json` | _未記入_ |

## Phase 1 後に確定する項目

以下は Phase 1 完了時点で記入し、別コミット（タグ `phase1-complete`）で固定する。

| 項目 | 値 |
|---|---|
| 実測 mixed プロンプト率 | _未記入_ |
| mixed 内陽性率 | _未記入_ |
| 逆算プロンプト数 M | _未記入_ |
| データ分割定義 `configs/data_splits.json` の SHA-256 | _未記入_ |
| アブレーション用プロンプト ID `configs/ablation_prompt_ids.json` の SHA-256 | _未記入_ |

> `data_splits.json` と `ablation_prompt_ids.json` は Phase 2 の**生成開始前**に確定・ハッシュ記録すること（計画書 §2.3 / §2.4）。

## PCA 診断値（Phase 0 で記録）

| 項目 | 値 |
|---|---|
| PC1–64 累積寄与率 | _未記入_ |
| PC1–128 累積寄与率 | _未記入_ |
| PC1–3 の寄与率合計 | _未記入_ |

## ハッシュ算出コマンド

```bash
find configs src docs -type f \( -name '*.py' -o -name '*.json' -o -name '*.npy' -o -name '*.md' \) \
  | sort | xargs shasum -a 256
```
