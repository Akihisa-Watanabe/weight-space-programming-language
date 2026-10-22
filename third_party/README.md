# third_party

関連する研究の参照実装（git submodule）と論文（[papers/](papers/README.md)）を置いています。設計の中での位置づけは [docs/architecture.md](../docs/architecture.md)、各論文の内容は [docs/related_works.md](../docs/related_works.md) を参照してください。

## 使い方

```sh
git submodule update --init                       # 登録済みの submodule をすべて取得（浅い clone）
git submodule update --init third_party/SEAL      # 1件だけ取得
git submodule status                              # 固定しているコミットを確認（先頭が "-" なら未取得）
git submodule update --remote third_party/rome    # 上流の最新に更新（固定コミットが変わるのでコミットが必要）
```

`.gitmodules` で `shallow = true` を指定しているので、どれも履歴を含まない浅い clone になります。

## 方針

- コア（`src/weightir`）は、ここにあるコードを import しません。ここのコードは、再現（元の実装と数値が一致するか）と比較の基準として、評価（`evals/`）から使います。
- ライセンスが付いていないリポジトリ（`unified-model-editing`、`task_vectors`、`composable-interventions`）は、読むだけにしてコードを流用しません。
- 作業ツリーが大きい4件（`RippleEdits`、`MQuAKE`、`APRNN`、`SEAL`）は、登録してコミットを固定しただけで、まだ取得していません。必要になったら上のコマンドで取得してください。
- ライセンスは、GitHub が判定したもの（2026年10月時点）です。流用する前に、各リポジトリの LICENSE を確認してください。

## 一覧

### 閉形式の重み編集と、その評価

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `EasyEdit` | [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) | 編集手法と評価データをまとめた枠組み。ROME・MEMIT・AlphaEdit・MEND・MALMEN などの比較と、CounterFact・zsRE での評価 | MIT | 取得済み |
| `rome` | [kmeng01/rome](https://github.com/kmeng01/rome) | ROME の元実装。キー共分散 C の推定と、k*・v* の作り方の基準 | MIT | 取得済み |
| `memit` | [kmeng01/memit](https://github.com/kmeng01/memit) | MEMIT の元実装。複数層への分配と、層ごとのキーの再計算 | MIT | 取得済み |
| `AlphaEdit` | [jianghoucheng/AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) | 零空間射影の元実装（閾値の扱いの確認） | MIT | 取得済み |
| `unified-model-editing` | [scalable-model-editing/unified-model-editing](https://github.com/scalable-model-editing/unified-model-editing) | EMMET（等式制約つきのバッチ編集） | なし | 取得済み |
| `RippleEdits` | [edenbiran/RippleEdits](https://github.com/edenbiran/RippleEdits) | 編集の波及を測る評価データ | MIT | 未取得（約120MB） |
| `MQuAKE` | [princeton-nlp/MQuAKE](https://github.com/princeton-nlp/MQuAKE) | 多段推論を測る評価データ | MIT | 未取得（約90MB） |

### プログラム ⇄ Transformer の重み

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `tracr` | [google-deepmind/tracr](https://github.com/google-deepmind/tracr) | RASP を重みに変換するコンパイラ。中間表現 Craft の設計の参考と、正解の部分空間が分かるテストベッド（更新停止） | Apache-2.0 | 取得済み |
| `RASP` | [tech-srl/RASP](https://github.com/tech-srl/RASP) | RASP のインタプリタ | MIT | 取得済み |
| `alta` | [google-deepmind/alta](https://github.com/google-deepmind/alta) | ループつきの言語とコンパイラ。層ごとの目標で学習させる方法（trace supervision） | Apache-2.0 | 取得済み |
| `InterpBench` | [FlyingPumba/InterpBench](https://github.com/FlyingPumba/InterpBench) | 既知の回路を実装したモデル群（テストベッド T1） | MIT | 取得済み |

### 重みの代数・マージ・重み操作言語

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `mergekit` | [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | YAML によるマージ手順の言語。比較対象、パッチの書き出し先の候補 | LGPL-3.0 | 取得済み |
| `brainsurgery` | [schneiderkamplab/brainsurgery](https://github.com/schneiderkamplab/brainsurgery) | 宣言的な重み操作のDSL。パッチの実行系の候補 | MIT | 取得済み |
| `evolutionary-model-merge` | [SakanaAI/evolutionary-model-merge](https://github.com/SakanaAI/evolutionary-model-merge) | マージ手順の進化的探索 | Apache-2.0 | 取得済み |
| `task_vectors` | [mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors) | Task Arithmetic | なし | 取得済み |
| `dataless-model-merging` | [bloomberg/dataless-model-merging](https://github.com/bloomberg/dataless-model-merging) | RegMean（最小二乗による閉形式の merge）（更新停止） | Apache-2.0 | 取得済み |
| `composable-interventions` | [hartvigsen-group/composable-interventions](https://github.com/hartvigsen-group/composable-interventions) | 編集・圧縮・アンラーニングの合成の評価プロトコル | なし | 取得済み |

### モデル内部へのアクセス

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `TransformerLens` | [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | フックによる活性値の取得 | MIT | 取得済み |
| `nnsight` | [ndif-team/nnsight](https://github.com/ndif-team/nnsight) | 介入グラフによる、遅延評価の介入 | MIT | 取得済み |
| `pyvene` | [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene) | 設定として宣言する介入 | Apache-2.0 | 取得済み |

### 仕様・検証・保証つきの介入

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `PRDNN` | [95616ARG/PRDNN](https://github.com/95616ARG/PRDNN) | 1層の線形計画による、証明つきの修復 | MIT | 取得済み |
| `APRNN` | [95616ARG/APRNN](https://github.com/95616ARG/APRNN) | アーキテクチャを保つ、証明つきの修復 | 独自（GitHub で判定できず） | 未取得（約120MB） |
| `vehicle` | [vehicle-lang/vehicle](https://github.com/vehicle-lang/vehicle) | 型つきの仕様言語。型による拒否と、ハッシュによる由来管理の参考 | 独自（GitHub で判定できず） | 取得済み |
| `concept-erasure` | [EleutherAI/concept-erasure](https://github.com/EleutherAI/concept-erasure) | LEACE（閉形式で保証つきの概念の消去） | MIT | 取得済み |
| `refusal_direction` | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | 方向の除去を、重みの直交化として焼き込む | Apache-2.0 | 取得済み |

### LLM・探索による合成、学習済み最適化器

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `learned_optimization` | [google/learned_optimization](https://github.com/google/learned_optimization) | VeLO を含む学習済み最適化器（最適化プログラムの比較対象） | Apache-2.0 | 取得済み |
| `ShinkaEvolve` | [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | LLM による進化的なプログラム探索 | Apache-2.0 | 取得済み |
| `openevolve` | [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve) | AlphaEvolve のオープンな実装 | Apache-2.0 | 取得済み |
| `SEAL` | [Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL) | LLM が自己編集（学習データと設定）を書く、最も近い先行研究 | MIT | 未取得（約390MB） |
| `text-to-lora` | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) | 説明文から LoRA を作るハイパーネットワーク（比較対象） | Apache-2.0 | 取得済み |

### 制約つき学習と更新の幾何

| パス | 元のリポジトリ | このリポジトリでの使い道 | ライセンス | 状態 |
|---|---|---|---|---|
| `modula` | [modula-systems/modula](https://github.com/modula-systems/modula) | modular norm（層の種類ごとのノルムと双対化）の実装 | MIT | 取得済み |
| `Muon` | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | Muon 最適化器 | MIT | 取得済み |
| `GPM` | [sahagobinda/GPM](https://github.com/sahagobinda/GPM) | 勾配を保存部分空間の外へ射影する継続学習 | MIT | 取得済み |
| `Adam-NSCL` | [ShipengWang/Adam-NSCL](https://github.com/ShipengWang/Adam-NSCL) | Adam の後で零空間へ射影する継続学習（交換則の実例） | MIT | 取得済み |
| `InfLoRA` | [liangyanshuo/InfLoRA](https://github.com/liangyanshuo/InfLoRA) | 固定した基底による再パラメータ化で干渉を防ぐ LoRA | MIT | 取得済み |
