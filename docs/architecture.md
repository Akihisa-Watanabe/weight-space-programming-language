# アーキテクチャ

この文書は、このリポジトリのシステムの設計と、それが今どう実装されているかをまとめたものです。設計判断の根拠となる論文は [related_works.md](related_works.md)、実験の定義と結果は [experiments.md](experiments.md)、参照実装の一覧は [third_party/README.md](../third_party/README.md) にあります。

## 1. 何を作るか

LLMに重みの数値を出力させる代わりに、**重みを編集するプログラム**を書かせるシステムを作ります。プログラムは、線形代数で意味が決まる型つきの言語（作業名 **WeightIR**）で書き、Python に埋め込みます。コンパイラは、プログラムを1つの重み行列についての制約つき最適化問題に変換して閉形式で解き、元のデータで制約を測り直してから、次の2つを返します。

- **パッチ**：重みの差分 ΔW
- **検査の結果**：元のデータで測り直した残差、変化の大きさとランク、予算を満たしたか

同じ編集の言語を、2つの場面で使います。

| 場面 | 例 | 実装 |
|---|---|---|
| 一回きりの編集 | 「キー K に値 V を返し、部分空間 S への応答は変えない。変化量はC計量で最小、ランクは8以下」 | `wir.edit` → `wir.compile` → `Model.apply`。ROME などはプリセット（`weightir.presets`） |
| 学習（毎ステップの編集） | 「このバッチの入力を、損失の勾配に沿って動かした出力へ写す。変化量はC計量で最小」 | LLM が編集プログラム `step(ctx)` を書き、`ProgramOptimizer` がそれを毎ステップ実行する。LLM は試行（新しい初期値からの学習）の結果を見て書き直す（`agentic.loop`） |

2つ目が、リポジトリ名の agentic optimization にあたります。LLMは重みも更新則も直接は出力せず、毎ステップの重みの変化を編集として書きます。重みの変化はすべて、プログラムが出した編集の閉形式の解から来ます。同じ制約を固定の更新則にした `weightir.train.Rule` もあり、比較の基準と、LLMを使わない制約つきの学習に使います。

### 目標

- 編集で「何を変え、何を保存し、どの予算の中で変えるか」を型で書けるようにし、満たせない指定は理由（矛盾の証拠、必要なランク、最小のノルム）つきで拒否する。
- 既存の編集手法（ROME、MEMIT、EMMET、AlphaEdit）を、アルゴリズムとしてではなく、同じ言語で書いた仕様の具体例（プリセット）として表す。
- LLMが試行を重ねて編集プログラムを改善し、同じ試行回数で学習率を選んだ Adam より良く学習できるかを確かめる。
- 同じ数値計算部分を使ったとき、LLMが素の Python や JSON よりこの言語で良い編集を書けるかを実証する（主実験、未実装）。

### 当面は対象外

- 「能力A＋能力B」のような、能力についての完全な代数
- 異なるモデル間での編集の移植
- 証明支援系による形式証明（数値で測り直すところから始め、必要な中核だけを後で形式化する）
- 生成列全体についての保証

## 2. 設計原則

文献調査と検算から、次の原則を採ります。根拠は10節の表にまとめています。

1. **DSLは Python に埋め込む。** 独自構文の言語はLLMが書き損じやすいので、型つきの Python API として提供します。
2. **アルゴリズムではなく仕様を中心に据える。** ROME などは「仕様を組み立てる関数」（`presets`）として実装し、解き方（`solve`）とは分けます。
3. **計量（変化量の測り方）を明示する。** `minimize=Covariance(C)` を渡すと `tr(ΔW C ΔWᵀ)`、渡さないと `‖ΔW‖_F²` を最小にします。C計量は、入力のどんな可逆な座標変換でも解が変わりません。`‖ΔW‖_F` は直交変換以外では変わります。プリセットと例の編集プログラムは C計量を使います。
4. **ソルバと検査を分ける。** ソルバ（`solve`）は候補を出すだけです。検査（`check`）が、元の K・V・U（推定した部分空間では、推定に使ったデータ）に対して制約と予算を測り直します。保存する dtype（bfloat16 など）に丸めた後の残差も測ります。
5. **黙って緩めない。** 厳密な制約（`Exact`）と罰則（`Soft`）を型で分け、厳密な制約を満たせなければ `Infeasible` を送出します。閾値で推定した部分空間は `Subspace(exact=False)` として、厳密な部分空間と区別します。
6. **「不可能と証明した」と「判定できない」を分ける。** `Infeasible.kind` で、矛盾・ランク・ノルムの証明と、数値の許容誤差の中で判定を保留したものを区別します。
7. **コアを LLM にも torch にも依存させない。** `spec`・`solve`・`check`・`compile` は numpy と scipy だけを使います。LLMの層、プリセット、素の Python のエージェントが、同じバックエンドを共有できるようにするためです。比較実験で「言語の効果」と「ソルバの効果」を分けられます。
8. **参照実装は import せず、コピーして直す。** `third_party/` を消しても、学習と評価を含めてすべて動きます（CLAUDE.md）。

## 3. 全体構成

```
experiments/  分類・強化学習・知識編集の実験スクリプト、最適化器の一覧（optimizers.py）
   │
agentic/      LLMが編集プログラムを書く層
   llm        Claude の呼び出し（anthropic SDK）
   program    ProgramOptimizer（編集プログラムを毎ステップ動かす）
   loop       試行ループ（LLM がプログラムを書き、試行の結果を見て直す）
   │
evals/        CounterFact の指標、コーパス（wikitext-103）と統計量のキャッシュ
   │
weightir/     編集の言語とコンパイラ
   spec       DSL：Keys・Values・Subspace・Covariance、Exact・Soft、Rank・Norm、edit・joint・span・subspace
   solve      閉形式の求解、予算、実現できないことの判定（Infeasible）
   check      元データでの制約の測り直し
   compile    compile・then、Patch
   ─────────  ここまで numpy と scipy だけ
   model      サイトの規則（学習・編集できる Linear と Conv1D）と重みの読み書き、Hugging Face モデル
   nethook    フック（ROME から移植）
   stats      キー、入力の2次モーメント、文脈テンプレート、AlphaEdit の保存部分空間
   targets    目標値（ROME の v*、MEMIT の z）
   presets    ROME・EMMET・MEMIT・AlphaEdit
   train      学習用の Rule と Optimizer
```

依存の向きは上から下だけです。`weightir` は `agentic` も `evals` も import しません。どのモジュールも `third_party` を import しません。

## 4. 編集の言語（weightir.spec）

1つの編集は、1つの重み行列（サイト）についての次の問題です。W は向きによらず（出力次元, 入力次元）で扱います（GPT-2 の Conv1D は (入力, 出力) で保存しているので、`Model.weight` が転置します）。

```text
最小化  tr(ΔW M ΔWᵀ)                              M は minimize の C（なければ I）
      + Σ Soft の写像   w‖(W + ΔW)K − V‖²
      + Σ Soft の保存   w·tr(ΔW C_j ΔWᵀ)
      + ridge·‖ΔW‖²
制約    (W + ΔW)K = V    Exact の写像
        ΔW U = 0         Exact の保存（span(U) の入力への応答は変えない）
予算    rank(ΔW) ≤ r,  ‖ΔW‖ ≤ B
```

| 型 | 中身 |
|---|---|
| `Keys(data, site)` | 入力側のベクトル (in, p)。1列が1例 |
| `Values(data, site)` | キーに対応する出力の目標 (out, p) |
| `Subspace(basis, site, exact, source)` | 入力空間の部分空間の正規直交基底。`span` で作ると厳密（`exact=True`）、`subspace(..., energy / threshold / rank)` や `stats.top_eigenspace` で作ると推定（`exact=False`）。`source` に元のデータを持ち、`check` がそのデータでの出力の変化を測る |
| `Covariance(C, site, n)` | 入力の2次モーメント E[xxᵀ]（n 個のベクトルから推定） |
| `Exact(K, V)`、`Soft(K, V, weight=w)` | 写像 |
| `Exact(U)`、`Soft(C, weight=w)` | 保存 |
| `Rank(r)`、`Norm(B, kind="fro" \| "metric")` | 予算 |
| `Edit` | `edit(site, map, preserve, minimize, require, ridge)` が作る。すべての部品が同じサイトに属し、形が合うかを作るときに検査する（合わなければ `SpecError`） |
| `Patch(site, delta, check)` | `compile` の出力 |
| `Infeasible(kind, details)` | 解がないことの証明、または判定の保留（6節） |

`joint(*edits)` は、同じサイトの複数の編集を1つの問題にまとめます。

### 計量の選び方

座標の取り方の自由（ゲージ）は、層の入力空間ごとに違います。計量を選ぶときに効きます。

- **残差ストリーム**：RMSNorm でつながったモデルでは、直交回転しても関数が変わりません（SliceGPT）。`‖ΔW‖_F` は直交回転で変わらないので、ここでは安全です。
- **注意ヘッドの内部**（W_Q と W_K の間、W_V と W_O の間）：一般の可逆変換 GL(d_head) の自由があります（Tran et al. 2025）。W_Q 単体の `‖ΔW‖_F` 最小化は座標に依存するので、積 `W_Q W_Kᵀ` を対象にするか、C計量を使います。
- **SwiGLU の中間層**：ニューロンごとに拡大縮小する自由（対角行列）があります。`down_proj` の入力側で `‖ΔW‖_F` を使うと、座標に依存します。

C計量 `tr(ΔW C ΔWᵀ)` は、どんな可逆な座標変換でも解が変わらないことを確認済みです。ゲージを自動で検査する仕組みはありません。

## 5. 編集1件の流れ

GPT-2 XL での例です。ROME の問題（1件の厳密な写像、C計量）に、汎用テキストの入力が主にある部分空間の保存と、ランクの予算を加えています。

```python
import weightir as wir
from weightir import presets, stats, targets
from weightir.model import Model
from evals import data

m = Model.load("openai-community/gpt2-xl")
site = "transformer.h.17.mlp.c_proj"
hp = presets.HPARAMS["gpt2-xl"]["rome"]
templates = data.templates(m, hp["length_params"])                      # ROME の文脈テンプレート
req = targets.Request(prompt="{} plays the sport of", subject="LeBron James", target_new="football")

K = stats.keys(m, site, [req.prompt], [req.subject], templates)         # 主語の最後のトークンでの入力
V = targets.optimize_values(m, site, [req], templates, **hp["opt"])    # ROME の v*
C = data.covariance(m, site)                                            # wikitext-103 での入力の2次モーメント
S = wir.subspace(data.keep_inputs(m, site), site, energy=0.9)           # 汎用テキストの入力が主にある部分空間

e = wir.edit(site, map=wir.Exact(K, V), preserve=wir.Exact(S), minimize=C, require=[wir.Rank(1)])
patch = wir.compile(e, m, dtype="float32")   # 満たせなければ wir.Infeasible
m.apply(patch)                               # m.reset() で元に戻す
```

`presets.rome(m, req, C, site=site, templates=templates, **hp["opt"])` は、保存と予算のない同じ編集を返します。

`compile` の中では、次の順に処理します。

1. **組み立て**：`edit` が、型・サイト・形を検査します（`SpecError`）。`lower` が、W と形が合うかを確かめ、配列に直します。
2. **実現可能性の解析**（厳密な写像があるとき）：`Q = (I − UUᵀ)K` と `R = V − WK` を作り、`ΔW Q = R` に解があるかを判定します。必要なランク（= rank R）と、厳密な制約を満たす最小のノルムを、予算と比べます。
3. **求解**：閉形式で解きます（6節）。
4. **予算**：ノルムの予算を超えたら、罰則 η‖ΔW‖²（`kind="metric"` なら η·tr(ΔW M ΔWᵀ)）を足し、予算に収まる最小の η を二分探索します。ランクの予算は、ソフトな問題（厳密な写像なし）なら、目的関数の計量で最良のランク r の解に切り詰めます。
5. **検査**：元の K・V・U で残差を測ります。float64 と、指定した dtype に丸めた後の両方で測り、予算も確認します。

## 6. ソルバと診断（weightir.solve、weightir.check）

すべての編集は、次の形に帰着します。

```text
最小化  tr(ΔW H ΔWᵀ) − 2 tr(ΔW Gᵀ)    制約  ΔW A = B
A = [K_exact, U],  B = [V_exact − W K_exact, 0]
H = M + Σ_soft w K Kᵀ + Σ_soft w C + ridge·I,   G = Σ_soft w (V − W K) Kᵀ
```

H が正定値なら、解は一つで `ΔW = G H⁻¹ + Λ (H⁻¹A)ᵀ`、`Λ = (B − G H⁻¹ A)(Aᵀ H⁻¹ A)⁺` です。特別な場合として、

- ソフト項も計量もなければ `ΔW = B A⁺`（最小フロベニウスノルム）
- H = C なら `B (Aᵀ C⁻¹ A)⁺ Aᵀ C⁻¹`（ROME、EMMET の形）
- 重み1の `Soft` の写像と `minimize=λC` なら `R Kᵀ(λC + KKᵀ)⁻¹`（MEMIT の形）

になります。どの閉形式も、vec(ΔW) についての KKT 系を別に解いた結果と突き合わせてテストしています（`tests/test_solve.py`）。

### 既存手法との対応（weightir.presets）

| プリセット | 問題 | 元の手法との違い |
|---|---|---|
| `rome` | 1件の `Exact(k̄, v*)`、`minimize=C` | 制約にも平均キー k̄ を使う（r-ROME。元の実装は方向に k̄、制約に素のプロンプトのキーを使う） |
| `emmet` | バッチの `Exact(K, V)`、`minimize=C` | 論文に書かれた厳密な目的関数を解く（元の実装はバッチで αI を足すので、MEMIT に近くなる） |
| `memit` | 層ごとに `Soft(K, V)`、`minimize=λC` | 残差を層に分け、層ごとに途中まで編集したモデルでキーを計算し直す（元の実装と同じ） |
| `alphaedit` | 層ごとに `Soft(K, V)`、`Exact(U)`、過去の編集のキーの Gram を `Soft` で保存、リッジ | U は編集前の C の固有値が τ = 2e-2 を超える固有ベクトル（公開コードの値。論文の本文は1e-2） |

### 診断

| 診断 | 意味 | 主張の強さ |
|---|---|---|
| `Infeasible("conflict")` | 厳密な写像同士、または写像と保存が矛盾する。証拠として、`Qa ≈ 0` かつ `Ra ≠ 0` となる係数 a を返す | 代数的に不可能 |
| `Infeasible("rank")` | 厳密解はすべて、ランクが `required_rank` 以上 | 予算内の厳密解はない |
| `Infeasible("norm")` | 解はすべて、ノルムが `min_norm` 以上 | 予算内の解はない |
| `Infeasible("unresolved")` | 特異値が許容誤差の帯 (tol, 1e3·tol] にあり、判定できない | 判定を保留 |
| `Infeasible("unsupported")` | この組み合わせは強制できない（目的関数が正定値でない、厳密な写像とソフト項とランクの予算の組など） | 未対応 |
| `SpecError` | 型・サイト・形の誤り | — |

許容誤差は `tol = rtol·‖K‖₂`（rtol の既定は1e-10）で、Q の特異値のうち tol 以下は0とみなします。キーの大きさを基準にするので、保存する部分空間にほぼ含まれるキーも矛盾として検出できます。

`check` は辞書を返します：`map_residual`（厳密なキーでの相対残差の最大）、`soft_map_residual`、`preserve_residual`（‖ΔW U‖ / ‖ΔW‖）、`preserve_data_change`（推定した部分空間の元データでの出力の相対変化）、`soft_preserve_penalty`、`fro`、`rank`、`metric`、`budgets_ok`、`ok`。dtype を指定すると、丸めた後の `map_residual_<dtype>` と `preserve_change_<dtype>` も加えます。`ok` が False でも例外にはしません。

「不可能と証明した」と「この方法では判定できない」を分けるのは、出力レベルの修復が ReLU の1層だけでも NP 困難で、実用的な手法の多くが不完全だからです（PRDNN、APRNN、PREPARED）。

## 7. 合成

| 演算 | 意味 | 成り立つ法則（単一サイト・厳密制約・実現可能なとき） |
|---|---|---|
| `joint(A, B)` | 制約をまとめ、一つの問題として解く | — |
| `then(model, A, build_B)` | A を適用し、編集後のモデルを `build_B` に渡して B を組み立てて解く。同じサイトなら、A の厳密なキーを `Exact(span(K_A))` として保存に加える（`keep_previous=True`、既定） | `then(A, B) = joint(A, B)`（テスト済み） |
| `then(..., keep_previous=False)` | A の結果を保存しない | 一般に順序に依存する |
| `merge`（作っていない） | 同じ基準から作った2つのパッチの和 | 両方の仕様を満たす ⇔ `ΔW_B K_A = 0` かつ `ΔW_A K_B = 0`。このとき `joint(A, B)` と一致 |

C計量のときは、白色化した座標で同じ法則が成り立ちます。干渉がない条件は `K_Aᵀ C⁻¹ K_B = 0` です。LLM内部のキーはこの意味で直交しないと報告されているので（Knowledge in Superposition）、`merge` は作らず、`joint` と `then` を使います。

複数サイトにまたがる編集では、A の書き込み先より下流の層で作ったキーや統計量は、A の後では古くなります。`then` は `build_B` に編集後のモデルを渡すので、そこで作り直します（MEMIT が層ごとにキーを計算し直すのと同じ考え方）。古い統計量を自動で検出する仕組みは作りません。この場合は上の法則が成り立たないので、合成後のモデルで評価し直します。

## 8. 学習

学習は3つの階層でできています。

| 階層 | 最適化されるもの | 動かすもの | その階層の設定 |
|---|---|---|---|
| 内側 | 重み | 編集プログラム（毎ステップ） | プログラムの中の歩幅など（LLM が書く） |
| 外側 | 編集プログラム | LLM（試行と試行の間） | LLM に送る文面、報告に入れる測定値、試行の回数・長さ・シード、モデルと effort、試行の時間の上限 |
| 実験 | — | 私たち | — |

外側の設定は、外側のループにとっての学習率にあたります。結果を報告する課題のテスト成績を見ながら直すことはせず、先に決めて固定します。変えたら全課題を回し直します。

### 8.1 ProgramOptimizer（agentic.program）

```python
optimizer = ProgramOptimizer(model, program=code)    # code: step(ctx) を定義する Python のソース
for x, y in data:
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    optimizer.observe(loss.item(), val_acc=...)       # 損失以外の測定値は、測ったステップでだけ渡す
```

- **サイト**：学習する Linear と Conv1D の層（他のパラメータと共有していないもの）。バイアスは重みと一緒に `[W | b]` として編集し、キーの最後の行を1にします。それ以外のパラメータ（正規化層、埋め込み、畳み込みの核）は学習せず、警告で名前を出します。
- **毎ステップ**：`step(closure)` は closure（前向きと逆向きの計算）を呼び、前向き計算のフックが、層の入力・出力・出力に対する勾配を記録します。続いてプログラムの `step(ctx)` を呼びます。プログラムは `ctx.apply(wir.edit(...))` でだけ重みを変えます。何も適用しなければ、重みは変わりません。
- **残すもの**：プログラムが `print` した内容の最後の部分（`output`）と、層ごとの編集の回数・`‖ΔW‖_F` の合計・写像の残差の合計（`edit_stats`、取り消した編集は数えない）。
- **エラー**：プログラムの例外は `ProgramError`（ステップ番号と、プログラム自身のフレームだけのトレースバック）として送出し、学習を止めます。`deadline`（試行の時間の上限）を渡すと、その時刻にプログラムの途中でも止めて、同じく `ProgramError` にします。止める合図は `Exception` ではないので、プログラムの `except Exception` では握りつぶせません。

プログラムが受け取る `ctx`：

| 属性・メソッド | 中身 |
|---|---|
| `ctx.sites`、`ctx.shape(site)` | このステップで学習の前向き計算をした層、(out, in)。in はバイアスの列を含む |
| `ctx.step`、`ctx.losses`、`ctx.metrics` | ステップ番号（1から）、`observe` で渡された損失のすべて、それ以外の測定値（名前ごとの (ステップ, 値) の列） |
| `ctx.keys(site)` | このステップの層の入力 (in, N)。1列がバッチの1行（系列ならトークン） |
| `ctx.outputs(site)`、`ctx.output_grads(site)` | 層の出力と、出力に対する損失の勾配 (out, N) |
| `ctx.values(site, step)` | `outputs − step·N·output_grads`。損失がバッチの平均なら、各例の出力を自分の損失の勾配に逆らって動かしたもの |
| `ctx.covariance(site, damping=1.0)` | 入力の2次モーメントの移動平均（減衰0.95）に `damping·mean(diag)·I` を足したもの |
| `ctx.weight(site)`、`ctx.apply(edit)` | 今の `[W \| b]`、編集を解いてすぐ書き込み、`Patch` を返す |
| `ctx.undo(patch)` | `ctx.apply` が返した編集を取り消す |
| `ctx.refresh()` | 今の重みで、このステップのバッチの前向きと逆向きの計算をやり直し、キー・出力・勾配を新しくして損失を返す |
| `ctx.state` | ステップをまたいで中身が残る辞書 |

例のプログラム（`EXAMPLE_PROGRAM`）は、各層でキーから `values(site, step=0.3)` への重み 1/N の `Soft` の写像を、`covariance(site)` を最小化して解きます。これは、K-FAC の入力側の因子で前処理した、学習率およそ0.3の勾配法の1ステップにあたります。どの層を、どの例で、どの制約と予算で編集するかは、プログラムが決めます。

プログラムは同じプロセスの中で `exec` します（名前空間は `wir` と `np`）。書くのは自分で選んだモデルで、敵対的な入力ではないので、隔離はしません。時間制限はメインスレッドでだけ効きます。

### 8.2 試行ループ（agentic.loop、experiments/search.py）

コーディングエージェントのループと同じ形です。LLM がプログラム（コード）を書き、ハーネスが試行（テスト）として新しい初期値から1本学習させ、報告を返し、LLM が直します。

```python
result = improve(task, sites, run, chat, trials=20)   # run(code) -> {"score", "summary", "report", "seconds", ...}
```

- **試行**：`run(code)` は、決めたシード（試行用で、評価には使わない）から、本番と同じ長さの学習を `ProgramOptimizer` で1本走らせます。試行どうしは独立なので、悪いプログラムが損をするのはその試行だけです。時間の上限（分類300秒、CartPole 120秒。例のプログラムの約7倍）を超えた試行は止めて、失敗として扱います。
- **LLM に送るもの**：最初はタスクの説明（課題、測定値、スコアの決め方、時間の上限）と層の形。試行のたびに、その試行の報告と、全試行の成績表、残りの試行数を送ります。報告に入るのは、検証の成績（分類）かエピソードのリターン（CartPole）の推移、学習損失の推移、時間、層ごとの編集の統計、プログラムが `print` した内容の最後の30行、またはエラーです。テストの成績は入れません。
- **ルール**：プログラムが使えるのは、`ctx` が渡すもの（そのステップのバッチ、重み、これまでの記録）だけです。新しい入力を作ること（データ拡張）と、ほかの場所のデータを読むことは禁止し、改善は重みの編集のしかたで出させます。目標値の作り方、どの層・どの例をどう編集するか、1ステップの編集の回数、制約・予算・平均化は自由です。禁止はシステムプロンプトで伝えるだけなので、結果はプログラムを読んで確かめます。
- **LLM の返事**：変える内容と理由を1〜2文、続けてプログラム全体の ```python ブロック。ブロックがない、出力が上限で切れた、読み込めない、のいずれかなら同じ会話で頼み直します（3回まで）。
- **選び方**：スコアが最も高い試行（同じなら速い方）のプログラムを残します。
- **最終評価**：残したプログラムを評価用のシード（分類 0〜2、CartPole 0〜4）で学習させ、ほかの最適化器と同じ形で `agent-<モデル>`（例：`agent-haiku-5-5`）として保存します。
- **公平な比較**：Adam にも同じ回数の試行を許し、学習率を 1e-4〜1e-1 の対数で等間隔に試して同じスコアで選び、同じ評価用のシードで `adam-tuned` として保存します。
- **LLM**：`--model` で選びます（既定は `claude-haiku-5-5`。プログラムを書くだけの役なので、Opus 5.5 の約40分の1の値段のモデルから試す）。effort high、出力の上限16,000トークン。システムプロンプトと、伸びていく会話をキャッシュします。サーバー側のフォールバックは使いません（実験の途中でモデルが入れ替わるため）。
- **記録**：`data/runs/search/<課題>-llm-<モデル>.json`（Adam は `<課題>-adam-lr.json`）に、全試行（プログラム、LLM が書いた理由、トークン数、スコア、報告、学習の記録）、会話の全文、トークン数の合計を残します。

LLM に更新則ではなく編集を書かせるのは、「LLMは重みを操作する代わりにプログラムを書く」を学習でも成り立たせるためです。毎ステップの重みの変化は、すべて型つきの編集で、コンパイラが解いたものです。

1本の学習の途中で LLM が書き直す形（以前の AgentOptimizer）も試しましたが、プログラムを試す場所が本番の学習しかなく、悪い変更を元に戻せないため、LLM は歩幅の調整しかしませんでした（[experiments.md](experiments.md) 5節）。そのため、試行と試行の間で書き直す形にしています。

### 8.3 固定の更新則：Rule（weightir.train）

`Optimizer(model, Rule(...))` は、torch の最適化器と1行で差し替えられます。各ステップで、各重み行列について次の問題を解きます。

```text
ΔW = argmin ⟨D, ΔW⟩ + tr(ΔW M ΔWᵀ) / (2·lr)   s.t.  ΔW U = 0,  ‖ΔW‖_M ≤ max_norm,  rank(ΔW) ≤ rank
```

- D は `direction` で決まります：sgd、momentum、adam、sign（Lion）、orth（モメンタムの Newton–Schulz による直交化、Muon）。
- M は `metric` で決まります：identity、または covariance（層の入力の2次モーメントの移動平均。K-FAC の入力側の因子）。covariance では `M = C + damping·mean(diag C)·I` で、damping の既定は1.0です。小さくすると（1e-3）、分散の小さい入力方向が増幅され、MNIST で学習が発散しました。
- 保存（`preserve`）の射影は、方向の変換の後で行います（Adam-NSCL と同じ）。重み減衰は計量と予算の外で分離して足し、同じ射影をかけます：`−lr·weight_decay·W(I − UUᵀ)`。
- 保存するサイトの重みは、前のステップの丸め残り r を足して `W + ΔW + r` を一度だけ丸めて書きます。丸め誤差が span(U) にたまりません。バイアスは凍結が必要です（バイアスの更新は span(U) の入力への出力も動かすため）。
- どの規則にも当てはまらない重み、共有された重み、バイアス、正規化層、埋め込みは、要素ごとの `default` 規則（既定は Adam）で更新します。

射影の順番が要るのは、保存空間へ射影した勾配 `GP` を各最適化器の変換に通すと、次の結果になったからです。

| 変換 | `ΔW U = 0` は保たれるか | 対応する手法 |
|---|---|---|
| SGD | 保たれる | OGD、GPM |
| Muon（Newton–Schulz による直交化）、polar(GP) | 保たれる | modular duality |
| Adam、符号化（要素ごとの変換） | 壊れる（この実験では更新の14〜45%が漏れた） | Adam の後で射影する（Adam-NSCL） |
| 分離型の重み減衰（−λW） | 壊れる（6%） | 減衰項も射影する |
| 固定した基底による再パラメータ化 | 最適化器によらず保たれる | InfLoRA |

`Rule` は最後の更新を射影するので、どの方向でも保存は保たれます。

## 9. エージェント層：これから作るもの

学習の試行ループのほかに、知識編集などで LLM が編集を書くエージェントを作ります。

- **入力**：課題の仕様（目標、D_edit、D_keep、予算、対象モデル）
- **出力**：WeightIR のプログラム
- **道具**：モデルの調査（サイトの一覧）、統計量の収集、試しのコンパイル（診断だけを返す）、評価、部品ライブラリ
- **ループ**：提案 → コンパイル → 診断 → 修正。LLMのトークン、コンパイル、評価の回数を予算として数えます。
- **比較条件**：すべて同じ weightir バックエンドを使います（[experiments.md](experiments.md) 8節）。
  1. LLM＋素の Python（型や検査なし、同じ数値関数）
  2. LLM＋JSON の仕様
  3. LLM＋型つきの埋め込みDSL
  4. ルールベースのプランナー＋型つきDSL
  5. 進化的探索（ShinkaEvolve、OpenEvolve）＋Python または DSL
- **部品ライブラリ**：よく使われる部分プログラムを、名前と説明つきの部品として登録します。LILO では、名前と説明のない部品は LLM の成績をかえって下げました。

上位層の仕事は、「D_edit で改善し、D_keep の振る舞いは保つ」という仕様から K・V・S を作ることです。

- **K（キー）**：編集したい入力について、選んだ位置（主語の最後のトークンなど）の活性値を集めます（`stats.keys`、`stats.collect`）。
- **V（目標値）**：多くの場合、最適化で求めます（`targets.optimize_values` は ROME の v*、`targets.optimize_residual` は MEMIT の z）。
- **S（保存する部分空間）**：D_keep の活性値から推定します（`wir.subspace`、`stats.top_eigenspace`）。推定は閾値つきの近似なので `exact=False` になります。
- **どのサイトを編集するか**：探索の対象として扱います。局在の解析結果は候補を出すのに使い、成否は評価で判断します（Hase et al. 2023、Makelov et al. 2024）。

## 10. 設計判断と根拠

| 判断 | 根拠 |
|---|---|
| DSLは Python に埋め込む | TracrBench：GPT-4-turbo が正しく書けた RASP は101問中57問、Python では約96%が正解。LILO：名前と説明のない部品は LLM の成績を下げる |
| 仕様を中心に据え、既存手法はプリセットにする | ROME・MEMIT・EMMET・AlphaEdit は同じ仕様の仲間（EMMET 論文、および検算）。プリセットのテストで、各手法の閉形式と一致 |
| 計量を明示し、C計量を勧める | 検算：‖ΔW‖_F の解は座標に依存し、C計量は依存しない。Park et al.（ユークリッド内積は自明な選択ではない）、Tran et al.（注意ヘッドの GL ゲージ） |
| ソルバと検査を分け、`Infeasible` で主張の強さを分ける | PRDNN・APRNN・PREPARED（不完全な手法）、Cordeiro et al.（実数上の証明が浮動小数点で破れる） |
| 保存する dtype に丸めた後も測る | Cordeiro et al. |
| `then` は既定で過去の編集を保存する | 検算（`then = joint`）、Knowledge in Superposition（順序に依存する干渉）、Composable Interventions |
| `merge` は作らない | 検算（`merge = joint` ⇔ 干渉項がゼロ）、Knowledge in Superposition（キーは直交しない） |
| 閾値つき部分空間は `exact=False` で区別する | AlphaEdit（論文は固有値1e-2、公開コードは平均した2次モーメントに対して2e-2）、その再現研究、BetaEdit、FisherEdit |
| 学習では方向の変換の後で射影し、重み減衰も射影する | 8.2節の検算、Adam-NSCL、modular duality、InfLoRA |
| 共分散計量の damping は mean(diag C) の1倍 | MNIST：1e-3 では発散した |
| LLM は試行と試行の間で書き直す（1本の学習の途中では書き直さない） | 途中で書き直す形では、悪い変更を元に戻せないため、LLM は歩幅の調整しかしなかった。コーディングエージェントと同じく、失敗しても捨てられる試行が要る |
| 学習中も LLM には更新則ではなく編集を書かせる | 「LLMは重みを操作する代わりにプログラムを書く」を学習でも成り立たせるため。OPTScientist や OPRO は更新則や解そのものを書かせる |
| サーバー側のフォールバックを使わない | 実験の途中で、評価しているモデルが入れ替わってしまう |
| コアを LLM に依存させない | 近い先行研究（Self-Developing、OPTScientist）には、同じバックエンドでの言語の比較がない |
| 参照実装は import せずコピーする | CLAUDE.md。ライセンス表示は [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |

以前の設計にあった、マニフェスト、由来と陳腐化の追跡、内容のハッシュによる同一性、保証の4段階（代数的・数値的・統計的・行動評価）は作りません（CLAUDE.md）。検査の結果は `Patch.check` の数値で、行動の評価は `evals` の指標で報告します。

## 11. マイルストーン

| 段階 | 内容 | 完了の目安 | 状態 |
|---|---|---|---|
| M0 | 数学カーネルと性質テスト | 閉形式・合成則・診断が、独立に解いた KKT 系と一致する | 済み |
| M1 | 実モデルでの単一サイトの編集とプリセット | ROME・MEMIT・EMMET・AlphaEdit をプリセットで再現し、元の実装の数値と一致する | 実装済み。小さなモデルで各手法の閉形式と一致。実モデルでの再現は未実施 |
| M2 | 合成、複数サイト、逐次編集 | 1万件までの逐次編集で干渉を測れる | `joint`・`then` は済み。逐次編集の評価は未実施 |
| M3 | エージェント層と比較実験 | 9節の5条件を同じ予算で比較できる | 学習の試行ループは済み。編集のエージェントと比較条件は未着手 |
| M4 | Tracr / InterpBench のテストベッド | 正解の部分空間に当たった率を報告できる | 未着手 |
| M5 | 学習（Rule、ProgramOptimizer と試行ループ） | 分類・強化学習・制約つきファインチューニングで Adam・SGD・Muon と比べられる | 分類と CartPole は済み（[experiments.md](experiments.md)）。制約つきファインチューニングは未実施 |
| M6 | 出力レベルの仕様（最終層の線形計画）と統計的保証 | PRDNN / PREPARED 型の契約を最終層で扱える | 未着手 |

## 12. リポジトリ構成

```
agentic_optimization/
├── README.md
├── CLAUDE.md                エージェント向けの規則
├── THIRD_PARTY_NOTICES.md   移植したコードの著作権表示とライセンス
├── docs/
│   ├── architecture.md      この文書
│   ├── experiments.md       実験の定義と結果
│   └── related_works.md     関連研究
├── src/
│   ├── weightir/            編集の言語・コンパイラ・検査、モデルの読み書き、プリセット、学習則
│   └── agentic/             LLM の呼び出し、ProgramOptimizer、試行ループ
├── evals/                   CounterFact の指標、コーパスと統計量のキャッシュ
├── experiments/             分類・強化学習・知識編集の実験スクリプト
├── tests/                   テスト（data/hf の極小モデルを使う）
├── data/                    モデル・データセット・統計量・実験結果（git では追跡しない）
└── third_party/             参照実装（submodule）と論文の一覧（papers/）
```

`third_party` の使い方：

- どのモジュールも `third_party` を import しません。必要なコードはコピーして直し、出典とライセンスを `THIRD_PARTY_NOTICES.md` とファイルの先頭に書きます。例外は、移植したコードと元の実装を比べる一回きりの確認だけです。
- 実行時に必要なライブラリ（torch、transformers など）は、パッケージとして依存します。
- ライセンスのないリポジトリ（`unified-model-editing`、`task_vectors`、`composable-interventions`）は読むだけにし、コードを流用しません。

## 13. 未解決の問題

- 数学能力や日本語能力のような広い能力を、部分空間に結びつける方法。非線形な対応づけを許すと、どんなネットワークもどんなアルゴリズムに対応づけられてしまう（Sutter et al. 2025）ので、対応づけの写像のクラスを型で制限する必要がある。
- 多層編集で、モデル全体について保証を得る方法。今のところ、Transformer でモデル全体の保証が得られた例は、最終層か出力層の編集に限られる。
- 浮動小数点での保証。
- LLMがこの言語を使いこなせるか。学習では、Haiku 5.5 が20回の試行で選んだプログラムが例のプログラムより良かったが、変えたのは主に歩幅の決め方で、編集の言語ならではの機能（厳密な写像、部分空間の保存、予算）はほとんど使われていない。
- 計算の重さ。ProgramOptimizer は毎ステップ、層ごとに float64 の連立方程式を CPU で解くので、Adam の数十倍遅い。入力次元 n の層は n×n の共分散を持つので、大きなモデルではメモリも要る。
- 探索と評価のコストを、公平に計上する方法。
