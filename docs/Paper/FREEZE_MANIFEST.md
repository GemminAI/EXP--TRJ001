# FREEZE MANIFEST — EXP-2026-NVS-001 v1.4

> ⚠️ **2026-09-14 追記**: 本ファイルが row #1 で参照する `docs/EXP-2026-NVS-001_v1.4.md` は、
> 本ファイルが記録する Phase 0〜3 実行（2026-09-13）より**後**の 2026-09-14 に、事後再構成
> 文書として起草された（`docs/EXP-2026-NVS-001_v1.4.md` 冒頭の警告を参照）。したがって下記の
> 実測値・Gate 3 判定は事前登録の確証的（confirmatory）結果としては扱えず、探索的
> （exploratory）として報告すること（原稿 `docs/EXP-TRJ001_Final_Manuscript_v3.4.md` §3.1a）。
> また下記の凍結情報テーブルが示す通り、`v1.4-frozen` タグ自体もまだ付与されていない。

本ファイルの全ハッシュ欄が埋まり、そのコミットに `v1.4-frozen` タグが付与された時点で凍結成立とする。
**凍結成立前に生成されたデータは本実験の結果として採用しない。**

## 凍結情報

| 項目 | 値 |
|---|---|
| 凍結日時 (UTC) | _未記入_（タグ付与時に記入） |
| 凍結コミット | _未記入_ |
| タグ | `v1.4-frozen` |
| 承認者 | _未記入_ |

## 環境（実測、`results/env_info.json`）

| 項目 | 値 |
|---|---|
| モデル | `Qwen/Qwen2.5-7B-Instruct`（U8：`meta-llama/Meta-Llama-3.1-8B-Instruct` から変更） |
| モデルコミットハッシュ | `a09a35458c702b33eeacc393d103063234e8bc28` |
| vLLM バージョン | `0.29.0` |
| CUDA / ドライバ | CUDA 13.0（torch `2.13.0+cu130`） |
| GPU | NVIDIA A40 |
| Python | 3.11.10 |

## 成果物ハッシュ（SHA-256、2026-09-13 実測）

| # | 成果物 | パス | SHA-256 |
|---|---|---|---|
| 1 | 実験計画書 | `docs/EXP-2026-NVS-001_v1.4.md` | `d37d3bc47e5520be628b279ca7d7407ecc0f99a277f9a0eb7bc60b25cf5e0910` |
| 2 | 射影行列 W（3584×128、U8：次元変更） | `configs/W_pca128.npy` | `dbfe086fa5dbfc965f28d78d072f2a6565752c1e19db8a5646403031656e285e` |
| 3 | 中心化ベクトル μ（3584） | `configs/mu_pca.npy` | `521eec467645365eb9e3c9a144b751a172ea504ab3acf4f2affed46f39b2d61b` |
| 4 | キャリブレーションプロンプト 200 件 | `configs/calibration_prompts_v1.json` | `492b3a64306bd0f197fa8f75e0d3391ac260d147ba32e7732e6983e14ba097e7` |
| 5 | Type A Phase 1 用プロンプト 1,000 件（v1.4 記述前提型） | `configs/prompts_v1.json` | `a86d476d395ddb6762f2d862515778106418b4daec3b149ac6fd5d2e39ba4f06` |
| 5b | Type A Phase 2 用プロンプト 493 件（U9、新規・独立在庫） | `configs/prompts_v1_phase2.json` | `85c6f92d0e9399ad0817099aa2d1429b3f99855813c0fa1d127aa8f8964882cf` |
| 6 | 棄権・ヘッジ判定パターン（U4：構文骨格パターン追加） | `configs/refusal_patterns_en.json` | `785419bf0cf5c227ebbae29b5ee0d997b20f89ffcdcc153ba014a7face488d94` |
| 7 | Type B 照合規則・正解集合 | `configs/typeB_reference.json` | `1c4257f0b9b0ed0a97bc3185f445363dbbf4578d64aea296bb9f1d266e0a86e8` |
| 8 | κ 計算モジュール | `src/compute_kappa.py` | `bebb8388913062c780d604c360ba8b93e7a23afcb3080d8e404870371d860f4b` |
| 9 | ラベル判定モジュール（v1.4：U1〜U4 反映） | `src/eval_hallucination.py` | `5dc9fd88969603ca0271e42dcbf857cc09fcd0c77dbae6b09305263f8b9fbd69` |
| 10 | 生成設定（U8：モデル・停止トークン変更） | `configs/generation_config.json` | `30a81eb5746d016c56a8c2feb47d9b5dbeec7915c6babfb6911b693f56e7d34d` |
| 11 | マスク手続き定義 | `configs/mask_protocol.json` | `ba08770073ce474cd4cfb43677751556a2989bb6583e634039d51e06e84bbd2c` |
| 12 | Type B 用プロンプト 200 件 | `configs/prompts_typeB_v1.json` | `caf7358d36a8022ed1f6db3efd8481a22994534b7cbec5622be3e1c32709d0fe` |

## Phase 1 完了時点で確定した項目（実測値、2026-09-13）

| 項目 | 値 |
|---|---|
| 実測 mixed プロンプト率 | 0.599（599/1,000 プロンプト） |
| mixed 内陽性率 | 0.471 |
| 逆算プロンプト数 M | 444.1 → 445（U9 により在庫 493 件を確保、追加生成なしで充足） |
| データ分割定義 `configs/data_splits.json` の SHA-256 | `ecd519da53f22d05dae5bbce0134a8dc9d9adbf8d78c27452a32497315f17b8f`（train 296 / val 99 / test 98、Phase 2 の 493 件プールに対する分割） |
| アブレーション用プロンプト ID `configs/ablation_prompt_ids.json` の SHA-256 | `78b39383a2e21592f52ed4ed99199131e172eda5793510b3629589e07736d355`（**98/200 件、U9 参照** — 目標未達をファイル内 `shortfall` フィールドに明記） |
| Phase 1 パイロット結果 | `results/phase1_pilot_trajectories.json`（1,000 プロンプト × 8 サンプル = 8,000 生成、Rule 3 **PASS**） |

> `data_splits.json` と `ablation_prompt_ids.json` は Phase 2 の生成開始前に確定・ハッシュ記録済み（本表）。

## PCA 診断値（Phase 0 実測、Qwen2.5-7B-Instruct、`configs/pca_diagnostics.json`）

| 項目 | 値 |
|---|---|
| PC1–64 累積寄与率 | 0.99992 |
| PC1–128 累積寄与率 | 0.99996 |
| PC1–3 の寄与率合計 | 0.99933 |
| 総サンプル数（トークン数） | 8,357 |

## Phase 2 完了時点で確定した項目（実測値、2026-09-13）

| 項目 | 値 |
|---|---|
| 採取件数 | 493 プロンプト × 8 サンプル = 3,944 生成 |
| ラベル内訳 | `type_a_positive`: 2,191 / `type_d_hedge`: 1,616 / `type_c_abstain`: 137 |
| 保存先 | `results/phase2_trajectories.json`（軽量メタデータ）＋ `results/phase2_projected/*.npz`（層別 128D 射影軌跡）＋ `results/ablation_raw_hidden_states/*.npy`（アブレーション対象 98 プロンプト分の生 3584D 隠れ状態） |

## Phase 3 完了時点で確定した項目（実測値、2026-09-13）

| 項目 | 値 |
|---|---|
| mixed プロンプト数（§5.5 フィルタ後） | 299 / 493（2,392 サンプル） |
| Layer 16 PROP-κ Test AUROC | 0.7806（95%CI [0.7298, 0.8232]） |
| Layer 16 BL-LEN Test AUROC | 0.8077（95%CI [0.7632, 0.8475]） |
| Layer 16 PROP-κ vs BL-LEN Δ | −0.0271（95%CI [−0.0727, 0.0190], p=0.869） |
| **Gate 3 条件 1**（AUROC>0.5, p<0.01） | **PASS**（CI 下限 0.730） |
| **Gate 3 条件 2**（Δ>0, p<0.05） | **FAIL**（CI 下限が 0 を含む） |
| **Gate 3 総合判定** | **FAIL** |
| F-Layer（Holm 補正、探索的） | Layer16 が Layer8・Layer24 いずれの PROP-κ AUROC も有意に上回る（p_holm < 0.001 両方） |
| ABL-1（生 3584D κ、Layer16、n=496） | AUROC 0.8472、vs BL-LEN Δ=+0.0395（95%CI [0.0114, 0.0726], p=0.004、**有意に上回る**） |
| ABL-2（PC4–67、Layer16、n=496） | AUROC 0.7969、vs BL-LEN Δ=−0.0108（95%CI [−0.0490, 0.0271], p=0.694、有意差なし） |
| 保存先 | `results/phase3_metrics.json` |

**Rule 1（§8）適用に関する注記**: 64D（PC1–64）の PROP-κ AUROC は 0.5 に近い値ではなく、単独では有意に 0.5 を上回る（Gate 3 条件 1 は PASS）。したがって §6.3 の「64D で AUROC≈0.5 だがアブレーションが有意」という分岐条件の前提（AUROC≈0.5）を厳密には満たさない。一方で ABL-1（生 4096/3584D 空間）は BL-LEN を有意に上回っており、64D への PCA 射影が応答長を超える κ 由来の情報の一部を喪失している可能性を示唆する、事前登録の分岐が想定していなかったパターンの結果である。これをどう解釈し次に進めるかは実験主宰者の判断に委ねる。
