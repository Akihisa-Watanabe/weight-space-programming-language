# アーキテクチャ

この文書は、このリポジトリで作るシステムの設計をまとめたものです。設計判断の根拠となる論文は [related_works.md](related_works.md) に、参照実装の一覧は [third_party/README.md](../third_party/README.md) にあります。現時点ではコードはなく、ここに書いた構成とAPIは提案です。

## 1. 何を作るか

LLMに重みの数値を出力させる代わりに、**重みを編集するプログラム**を書かせるシステムを作ります。プログラムは、線形代数で意味が決まる型つきの言語（作業名 **WeightIR**）で書きます。コンパイラはそれを制約つき最適化問題に変換して解き、独立に検証したうえで、次の2つを出力します。

- **パッチ**：重みの差分
- **マニフェスト**：何が、どの前提で保証されているかの記録

同じ言語で、2種類のプログラムを書けるようにします。

| 種類 | 例 | コンパイル先 |
|---|---|---|
| 編集プログラム（1回だけ実行） | 「キー K に値 V を返し、部分空間 S への応答は変えない。変化量は最小、ランクは8以下」 | 閉形式、または小さな最適化問題で解いたパッチ |
| 最適化プログラム（毎ステップ実行） | 「データ D で学習する。S への応答は変えない。更新のランクは8以下」 | 学習ループの各ステップで使う更新則（射影や再パラメータ化） |

2つ目が、リポジトリ名の agentic optimization につながります。LLMが最適化アルゴリズムを一度書けば、その後はLLMなしで毎ステップ実行できます。

### 目標

- 編集で「何を変え、何を保存し、その主張がどの前提で成り立つか」を、型とマニフェストで明示する。
- 既存の編集手法（ROME、MEMIT、EMMET、AlphaEdit）を、アルゴリズムとしてではなく、同じ言語で書いた仕様の具体例（プリセット）として表す。
- 同じ数値計算部分を使ったとき、LLMが素のPythonよりこの言語で良い編集を書けるかを実証する。

### 当面は対象外

- 「能力A＋能力B」のような、能力についての完全な代数
- 異なるモデル間での編集の移植
- 証明支援系による形式証明（数値検証から始め、必要な中核だけを後で形式化する）
- 生成列全体についての保証

## 2. 設計原則

文献調査と検算から、次の原則を採ります。根拠は11節の表にまとめています。

1. **DSLはPythonに埋め込む。** 独自構文の言語はLLMが書き損じやすいので、型つきのPython APIとして提供します。正規化したJSON表現は、記録・ハッシュ・比較実験に使います。
2. **アルゴリズムではなく仕様を中心に据える。** ROMEなどは「仕様を組み立てる関数」として実装し、解き方（ソルバ）とは分けます。
3. **計量（変化量の測り方）を必ず明示する。** 既定は、キーの共分散Cで重みづけた変化量 `tr(ΔW C ΔWᵀ)` です。これは座標の取り方に依存しません。`‖ΔW‖_F` も選べますが、座標に依存する場面では警告を出します。
4. **由来を型で追う。** キー・値・部分空間・共分散は、どのモデル状態の、どの層の、どのトークン位置の活性値から作ったかを持ちます。上流の重みが書き換わったら「古い」と判定します。
5. **ソルバと検証器を分ける。** ソルバは候補を出すだけです。検証器が、元のデータに対して制約を確かめ直します。
6. **黙って緩めない。** 厳密な制約を近似に置き換えるときは、新しい仕様として明示し、マニフェストに残します。
7. **保証は種類ごとに記録する。** 代数的・数値的・統計的・行動評価の4種類を、一つの `verified = true` にまとめません。
8. **コアはLLMに依存させない。** LLM層、ルールベースのプランナー、素のPythonエージェントが、同じバックエンドを共有できるようにします。比較実験で「言語の効果」と「ソルバの効果」を分けるためです。

## 3. 全体構成

```
agentic/     LLMプランナー・探索・部品ライブラリ
   │         課題 → プログラム（Python API / JSON）→ コンパイル → 診断 → 修正
   │         weightir の公開APIだけを使う
   ▼
weightir/    言語・コンパイラ・検証器（LLMに依存しない）
   lang      埋め込みDSL、正規化JSON
   passes    型検査 → 由来・陳腐化の検査 → ゲージ検査 → 実現可能性の解析 → 数学IRへの変換
   ir        LinearPatchProblem / MultiSiteProblem / OutputSpecProblem / TrainingConstraint
   solvers   exact_min_norm / exact_metric / soft_ridge / rank_constrained / lp_output
   optim     制約を学習則に変換（射影・再パラメータ化・交換則の管理）
   verify    独立検証（元データでの残差・予算・干渉、実際の数値形式での再検査）
   manifest  保証の記録、パッチの保存・適用・取り消し
   models    サイトとポートの解決、ゲージ、フック、重みの読み書き
   stats     K・V・C・S の収集と推定（由来つき、内容ハッシュでキャッシュ）
   targets   目標値 V の構成（ROME型の最適化など）
   ▼
evals/       テストベッド・比較手法・指標
third_party/ 参照実装とデータセット（submodule）、論文（papers/）
```

依存の向きは上から下だけです。`weightir` は `agentic` も `third_party` も import しません。`third_party` のコードは、再現と比較の基準として `evals` から使います。

## 4. 主な型

| 型 | 中身 | 由来として持つもの |
|---|---|---|
| `Snapshot` | モデルの重みへの不変な参照 | モデルID、revision、テンソルごとの内容ハッシュ |
| `Site` | 書き換え対象のパラメータ（例：`model.layers.12.mlp.down_proj.weight`） | 他の場所との共有（tied embeddingなど）、前後の正規化層 |
| `Port` | サイトの入力側または出力側のベクトル空間 | 次元、基底の性質、トークン位置の規則 |
| `Samples` | キーK、または値V（各列が1つの例） | Snapshot、Port、位置、データのハッシュ、件数、dtype |
| `Subspace` | 保存する部分空間の正規直交基底U | 推定法（エネルギー比・固有値の閾値・ランク）と閾値、元データ |
| `Metric` | 変化量の測り方（恒等、C、K-FAC型の G⊗C） | 推定に使ったデータと件数 |
| `Spec` / `Program` | 編集の仕様と、その合成（joint / then / merge） | 正規化JSONとそのハッシュ |
| `Problem` | 数学IR | 元の Spec |
| `Patch` | 差分（低ランク因子、または密行列） | 基準 Snapshot、書き込み先、読み取りの依存 |
| `Manifest` | 保証（Claim）と診断の一覧 | 上のすべてのハッシュ |

### ポートの基底とゲージ

`Port` は「どんな座標変換をしても関数が変わらないか」（ゲージ）を持ち、ゲージ検査に使います。

- **残差ストリーム**：RMSNormでつながったモデルでは、直交回転しても関数が変わりません（SliceGPT）。`‖ΔW‖_F` は直交回転で変わらないので、ここでは安全です。
- **注意ヘッドの内部**（W_Q と W_K の間、W_V と W_O の間）：一般の可逆変換 GL(d_head) の自由があります（Tran et al. 2025）。W_Q 単体の `‖ΔW‖_F` 最小化は座標依存になるので、積 `W_Q W_Kᵀ` を対象にするか、C計量を使います。
- **SwiGLUの中間層**：ニューロンごとに拡大縮小する自由（対角行列）があります。`down_proj` の入力側で `‖ΔW‖_F` を使うと、座標依存になります。

C計量 `tr(ΔW C ΔWᵀ)` は、どんな可逆な座標変換でも解が変わらないことを確認済みです。そのため既定にしています。

## 5. 編集1件の流れ

埋め込みDSLでの書き方の例です（APIは案）。

```python
import weightir as wir

m0 = wir.Snapshot.load("Qwen/Qwen2.5-0.5B", revision="...")
site = m0.site("model.layers.12.mlp.down_proj")

K = wir.collect(m0, site.input, prompts=EDIT_PROMPTS, position="last_subject_token")
V = wir.targets.optimize_values(m0, site, K, objective=NEW_FACTS)  # ROME型の v*
C = wir.second_moment(m0, site.input, corpus=GENERIC_TEXT, n=100_000)
S = wir.subspace(m0, site.input, data=KEEP_PROMPTS, rule=wir.Energy(0.99))

spec = wir.edit(
    site,
    map=wir.Exact(K, V),
    preserve=[wir.Exact(S)],
    minimize=wir.metric.covariance(C),
    require=[wir.Rank(le=8), wir.Norm(le=1.0)],
)
artifact = wir.compile(spec)  # Patch + Manifest、または診断
wir.check(artifact, numeric=wir.Policy(dtype="bfloat16", tol=1e-3))
```

`compile` の中では、次の順に処理します。

1. **型検査**：Kが `site.input`、Vが `site.output` のポートに属するか。KとVの列が同じ例に対応するか。
2. **由来・陳腐化の検査**：K・V・C・Sを作った Snapshot が、編集対象の Snapshot と一致するか。違う場合、上流の書き込みがこのポートに影響しうるなら `StaleRepresentation` を返します。
3. **ゲージ検査**：目的関数がポートのゲージで不変か。不変でなければ警告します。
4. **実現可能性の解析**：`Q = (I − UUᵀ)K` と `R = V − WK` を作り、`ker Q ⊆ ker R` を許容誤差つきで判定します。必要なランク（= rank R）と最小ノルムを、予算と比べます。
5. **数学IRへの変換**：目的関数・等式制約・ソフト項・予算・数値計算の方針を、`LinearPatchProblem` にまとめます。
6. **求解**：問題の形に合うソルバを選びます。
7. **独立検証**：元の K・V・U で残差を測ります。float64 と実際に使う dtype の両方で測り、予算も確認します。
8. **マニフェスト作成**：成り立った主張を、種類ごとに書きます。

### ソルバ（最初のセット）

表の記号：N は S⊥ の正規直交基底、A = NᵀK、M = NᵀCN です。

| ソルバ | 解く問題 | 閉形式 | 既存手法との対応 |
|---|---|---|---|
| `exact_min_norm` | min ‖ΔW‖_F s.t. ΔWK = R, ΔWU = 0 | ΔW = R Q⁺ | 構想の v0 |
| `exact_metric` | min tr(ΔW C ΔWᵀ) s.t. 同上 | ΔW = R (AᵀM⁻¹A)⁻¹ AᵀM⁻¹ Nᵀ | ROME（1件・Uなし）、EMMET（バッチ・Uなし） |
| `soft_ridge` | min ‖ΔWK − R‖² + λ tr(ΔW C ΔWᵀ) + μ‖ΔW‖² s.t. ΔWU = 0 | ΔW = R Aᵀ (AAᵀ + λM + μI)⁻¹ Nᵀ | MEMIT（Uなし・μ = 0）、AlphaEdit型（閾値つきU） |
| `rank_constrained` | 上の問題にランク上限を加えたもの | 縮小ランク回帰 | — |
| `lp_output`（後で） | 出力に対する線形制約（最終層のみ） | 線形計画 | PRDNN、PREPARED |

上の3つの閉形式は、別の解き方（ベクトル化したKKT系）と突き合わせて確認しました。差は1e-13以下です。`exact_metric` で U を空にすると EMMET の式に、さらに1件にすると ROME の式に一致します。

### 上位層：振る舞いの仕様から K・V・S を作る

「D_edit で改善し、D_keep の振る舞いは保つ」という仕様は、そのままでは解けません。K・V・S を作る必要があります。

- **K（キー）**：編集したい入力について、選んだ位置の活性値を集めます。位置の規則（主語の最後のトークンなど）も由来に含めます。
- **V（目標値）**：多くの場合、最適化で求めます（ROME の v*）。最適化の設定も由来に記録します。
- **S（保存する部分空間）**：D_keep の活性値から推定します。推定は閾値つきの近似になるので、閾値と元データを型に持たせます。
- **どのサイトを編集するか**：探索の対象として扱います。局在の解析結果は候補を出すのに使い、成否は検証で判断します（Hase et al. 2023、Makelov et al. 2024）。

LLMの主な仕事はこの層にあります。どのサイト・データ・位置・制約・合成を使うかを決め、診断を見て直します。

## 6. 合成の意味論

| 演算 | 意味 | 成り立つ法則（単一サイト・厳密制約・実現可能なとき） |
|---|---|---|
| `joint(A, B)` | 制約をまとめ、一つの問題として解く | — |
| `then(A, B)` | Aを適用したモデルでBを解く。既定でAのキーを厳密に保存する（`keep_previous=True`） | `then(A, B) = then(B, A) = joint(A, B)` |
| `then(A, B, keep_previous=False)` | Aの結果を保存しない | 一般に順序に依存する（反例あり） |
| `merge(P_A, P_B)` | 同じ基準から作ったパッチを足す。干渉の検査つき | 両方の仕様を満たす ⇔ `ΔW_B K_A = 0` かつ `ΔW_A K_B = 0`。このとき `joint(A, B)` と一致 |

どの法則も、短い証明があり、数値でも確認しました。C計量のときは、白色化した座標で同じ法則が成り立ちます。干渉がない条件は `K_Aᵀ C⁻¹ K_B = 0` になります。LLM内部のキーはこの意味で直交しないと報告されているので（Knowledge in Superposition）、`merge` は例外的な操作とし、既定は `joint` と `then` にします。

複数サイトにまたがる `then` では、Aの書き込み先より下流のポートで作った K・C・S を、すべて「古い」と判定します。そしてAを適用した後のモデルで作り直します（MEMIT が層ごとにキーを計算し直すのと同じ考え方）。この場合は上の法則が成り立たないので、合成後のモデルで検証し直します。

### 診断

| 診断 | 意味 | 主張の強さ |
|---|---|---|
| `ConstraintConflict` | 仕様が矛盾している（`Qa = 0` かつ `Ra ≠ 0` となる a を証拠として返す） | 代数的に不可能 |
| `NormBudgetExceeded` | 最小ノルム解でも予算を超える | 予算内の解は存在しない |
| `RankBudgetExceeded` | 必要なランク rank(R) が予算を超える | 予算内の厳密解は存在しない |
| `NumericallyUnresolved` | 特異値が閾値の近くにあり、可否を判定できない | 判定を保留 |
| `NotFound` | 不完全な手法（線形計画の緩和など）で解が見つからなかった | 不可能の証明ではない |
| `StaleRepresentation` | 統計量が古いモデル状態から作られている | — |
| `InterferenceDetected` | merge の干渉項がゼロでない（大きさも報告） | — |
| `GaugeDependentObjective`（警告） | 目的関数がポートのゲージで不変でない | — |
| `SharedParameter`（警告） | 書き込み先が他の場所でも使われている | — |
| `VerificationFailed` | 独立検証で制約を満たさなかった | — |

「不可能と証明した」と「この方法では見つからなかった」を分けるのは、出力レベルの修復が ReLU の1層だけでも NP 困難で、実用的な手法の多くが不完全だからです（PRDNN、APRNN、PREPARED）。

## 7. マニフェスト

パッチには、次のようなマニフェストを付けます（形式は案）。

```json
{
  "base_snapshot": "sha256:…",
  "writes": ["model.layers.12.mlp.down_proj.weight"],
  "reads": [{"port": "layers.12.mlp.down_proj:in", "snapshot": "sha256:…", "data": "sha256:…"}],
  "spec_hash": "sha256:…",
  "solver": {"name": "exact_metric", "version": "0.1.0"},
  "claims": [
    {"tier": "algebraic", "statement": "(W+ΔW)K = V and ΔW U = 0",
     "assumptions": ["exact arithmetic", "K, V, U computed on base_snapshot"]},
    {"tier": "numeric", "dtype": "bfloat16", "edit_residual": 3.1e-4, "keep_residual": 2.2e-5, "tol": 1e-3},
    {"tier": "statistical", "statement": "output change on held-out D_keep <= eps", "n": 2000, "confidence": 0.95},
    {"tier": "behavioral", "eval": "counterfact/efficacy", "value": 0.97}
  ],
  "scope": {"keys": "finite set of 64 examples", "preserve": "subspace S (energy 0.99 on D_keep)"},
  "not_claimed": ["effects through other ports", "preservation of whole generations", "properties after quantization"],
  "diagnostics": [{"kind": "GaugeDependentObjective", "severity": "warning"}]
}
```

- **代数的**：明示した前提のもとで、実数上の式として成り立つ。
- **数値的**：指定した dtype と許容誤差で、元データに対する残差が基準を満たした。量子化や回転（QuaRot）などの変換をした後は、測り直す。
- **統計的**：保留データ上の有限標本の保証（Provably Safe Model Updates の考え方）。
- **行動評価**：指定したデータと評価器で合格した。

キーの有限集合でだけ成り立つ保証と、部分空間全体で成り立つ保証は、`scope` で区別します。

## 8. 最適化プログラムへのコンパイル

同じ制約を、学習の更新則にも変換します。ただし、正しい変換は最適化器によって変わります。保存空間へ射影した勾配 `GP` を、各最適化器の変換に通したときの結果を確かめました。

| 変換 | `ΔW U = 0` は保たれるか | 対応する手法 |
|---|---|---|
| SGD | 保たれる | OGD、GPM |
| Muon（Newton–Schulz による直交化）、polar(GP) | 保たれる | modular duality |
| Adam、符号化（要素ごとの変換） | 壊れる（この実験では更新の14〜45%が漏れた） | Adamの後で射影する（Adam-NSCL） |
| 分離型の重み減衰（−λW） | 壊れる（6%） | 減衰項も射影する |
| 固定した基底による再パラメータ化 | 最適化器によらず保たれる | InfLoRA |

`optim` モジュールはこの交換則の表を持ちます。指定された最適化器と制約の組に対して、正しい順序の更新則を組み立てます。組み立てられなければ拒否します。計量は、modular norm（層の種類ごとにノルムを割り当てる理論）と、K-FAC型のC計量の両方を扱えるようにします。

これにより、LLMが「Dで学習し、Sは保存し、ランクは8以下、計量はC」という最適化プログラムを一度書けば、その後はLLMなしで毎ステップ実行できます。比較相手は AdamW、Muon、VeLO（`third_party/learned_optimization`）です。

## 9. エージェント層（agentic/）

- **入力**：課題の仕様（目標、D_edit、D_keep、予算、対象モデル）
- **出力**：WeightIR のプログラム
- **道具**：モデルの調査（サイトとポートの一覧）、統計量の収集、試しのコンパイル（診断だけを返す）、評価、部品ライブラリ
- **ループ**：提案 → コンパイル → 診断 → 修正。LLMのトークン、コンパイル、評価の回数を予算として数えます。
- **比較条件**：すべて同じ weightir バックエンドを使います。
  1. LLM＋素のPython（型や由来の検査なし、同じ数値関数）
  2. LLM＋JSONの仕様
  3. LLM＋型つきの埋め込みDSL
  4. ルールベースのプランナー＋型つきDSL
  5. 進化的探索（ShinkaEvolve、OpenEvolve）＋PythonまたはDSL
- **部品ライブラリ**：よく使われる部分プログラムを、名前と説明つきの部品として登録します。LILOでは、名前と説明のない部品はLLMの成績をかえって下げました。

## 10. 評価（evals/）

| 段階 | テストベッド | 確かめること |
|---|---|---|
| T0 | 合成した線形問題 | 数学カーネルの性質（正解が分かる） |
| T1 | Tracr のコンパイル済みモデル、InterpBench | 正解の部分空間が分かる環境で、プログラムが正しい部分空間に当たるか |
| T2 | 小型LLM（GPT-2、Qwen2.5-0.5B など）での事実の編集（CounterFact、zsRE） | ROME・MEMIT・EMMET・AlphaEdit をプリセットとして再現し、元の実装と数値が一致するか |
| T3 | RippleEdits、MQuAKE、汎用能力の評価、逐次編集（〜1万件）、圧縮との合成 | 層についての保証が、モデル全体でどこまで効くか |
| T4 | 制約つきファインチューニング | 最適化プログラム vs AdamW・Muon・VeLO |

**指標**
- 編集の成否：編集成功率、言い換えへの汎化、局所性、波及・多段推論、汎用能力
- 更新と制約：更新のノルムとランク、制約の残差
- 検査の性能：誤った仕様や古い前提を受理した率、陳腐化の検出率
- コスト：LLMのトークン、コンパイルと評価の回数、計算時間。シード数 × 反復回数のフロンティアとして報告します。

**評価の注意**
- teacher forcing ではなく、実際の生成で測ります。評価方法を見直すと、成功率が報告値の96.8%から38.5%に下がった例があります（Mirage of Model Editing）。
- エージェントが触れられない保留テストセットを使います。PostTrainBench では、エージェントがテストデータで学習してしまいました。

## 11. 設計判断と根拠

| 判断 | 根拠 |
|---|---|
| DSLはPythonに埋め込む | TracrBench：GPT-4-turbo が正しく書けた RASP は101問中57問、Python では約96%が正解。LILO：名前と説明のない部品はLLMの成績を下げる |
| 仕様を中心に据え、既存手法はプリセットにする | ROME・MEMIT・EMMET・AlphaEdit は同じ仕様の仲間（EMMET 論文、および検算） |
| 計量を明示し、既定はC計量にする | 検算：‖ΔW‖_F の解は座標依存、C計量は不変。Park et al.（ユークリッド内積は自明な選択ではない）、Tran et al.（注意ヘッドの GL ゲージ） |
| 由来と陳腐化を型で追う | MEMIT は層ごとにキーを計算し直す。AlphaEdit の射影は編集前のモデルで一度だけ計算。Vehicle は証明をファイルのハッシュに紐づける |
| ソルバと検証器を分け、診断で証明の強さを区別する | PRDNN・APRNN・PREPARED（不完全な手法）、Cordeiro et al.（実数上の証明が浮動小数点で破れる） |
| 保証を4種類に分ける | Provably Safe Model Updates（統計的保証）、Mirage of Model Editing（評価の落とし穴） |
| `then` は既定で過去の編集を保存する | 検算（`then = joint`）、Knowledge in Superposition（順序依存の干渉）、Composable Interventions |
| `merge` は干渉検査つきの例外にする | 検算（`merge = joint` ⇔ 干渉項ゼロ）、RegMean・MaTS（最小二乗による merge） |
| 制約と最適化器の交換則を管理する | 検算、Adam-NSCL、modular duality、InfLoRA |
| 閾値つき部分空間は「近似」と明記する | AlphaEdit（固有値1e-2以下）、その再現研究、BetaEdit、FisherEdit |
| コアをLLMに依存させない | 近い先行研究（Self-Developing、OPTScientist）には、同じバックエンドでの言語の比較がない |

## 12. マイルストーン

| 段階 | 内容 | 完了の目安 |
|---|---|---|
| M0 | 数学カーネルと性質テスト | `exact_min_norm`・`exact_metric`・`soft_ridge`・合成則・診断が、独立ソルバと一致する |
| M1 | 単一サイトの線形パッチを実モデルで（GPT-2 / Qwen2.5-0.5B）。型、由来つき統計量、検証器、マニフェスト | ROME・MEMIT・EMMET・AlphaEdit をプリセットで再現し、`third_party` の元実装と一致する |
| M2 | 合成（joint / then / merge）、陳腐化の検出、複数サイト、逐次編集 | 1万件までの逐次編集で、陳腐化と干渉を検出できる |
| M3 | エージェント層と比較実験 | 9節の5条件を同じ予算で比較できる |
| M4 | Tracr / InterpBench のテストベッド | 正解の部分空間に当たった率を報告できる |
| M5 | 最適化プログラムへのコンパイル | 制約つきファインチューニングで AdamW・Muon・VeLO と比較できる |
| M6 | 出力レベルの仕様（最終層の線形計画）と統計的保証 | PRDNN / PREPARED 型の契約を最終層で扱える |

## 13. リポジトリ構成（予定）

```
agentic_optimization/
├── README.md
├── docs/
│   ├── architecture.md      この文書
│   └── related_works.md     関連研究
├── src/
│   ├── weightir/            言語・コンパイラ・検証器（LLMに依存しない）
│   │   ├── lang/  passes/  ir/  solvers/  optim/
│   │   └── verify/  manifest/  models/  stats/  targets/
│   └── agentic/             LLMプランナー・探索・部品ライブラリ
├── evals/                   テストベッド・比較手法・指標
├── experiments/             実験の設定と結果
├── tests/                   単体テスト・性質テスト
└── third_party/             参照実装（submodule）と論文（papers/）
```

`third_party` の使い方：

- コア（`weightir`）は `third_party` を import しません。実行時に必要なライブラリ（torch、transformers など）は、パッケージとして依存します。
- `third_party` は、再現（元の実装と数値が一致するか）と比較の基準として `evals` から使います。
- ライセンスのないリポジトリ（`unified-model-editing`、`task_vectors`、`composable-interventions`）は読むだけにし、コードを流用しません。

## 14. 未解決の問題

- 数学能力や日本語能力のような広い能力を、部分空間に結びつける方法。非線形な対応づけを許すと、どんなネットワークもどんなアルゴリズムに対応づけられてしまう（Sutter et al. 2025）ので、対応づけの写像のクラスを型で制限する必要がある。
- 多層編集で、モデル全体について保証を得る方法。今のところ、Transformer でモデル全体の保証が得られた例は、最終層か出力層の編集に限られる。
- 浮動小数点での保証。
- LLMがこの言語を使いこなせるか。
- 探索と評価のコストを、公平に計上する方法。
