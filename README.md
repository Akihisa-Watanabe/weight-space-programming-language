# agentic_optimization

LLMに重みの数値を書かせる代わりに、**型つきの重み編集プログラム**を書かせる研究のリポジトリです。プログラムは Python に埋め込んだ DSL（作業名 WeightIR）で書きます。コンパイラは、それを1つの重み行列についての制約つき最適化問題に変換して閉形式で解き、元のデータで制約を測り直してから、重みの差分 ΔW を返します。

```text
最小化  tr(ΔW C ΔWᵀ)       変化量（C は層の入力の2次モーメント）
制約    (W + ΔW) K = V      キー K に値 V を返す
        ΔW U = 0            部分空間 span(U) の入力への応答は変えない
予算    rank(ΔW) ≤ r, ‖ΔW‖ ≤ B
```

LLMは「どの層の、どのキーを、どの値へ写し、何を保存するか」を書き、数値はコンパイラが計算します。ROME・MEMIT・EMMET・AlphaEdit は、この問題（またはその制約をゆるめた形）の具体例として表せます。同じ数学で、訓練の各ステップの更新則も書けます。

> 現状：DSL・ソルバ・実モデルの読み書き・プリセット・訓練用の最適化器・評価・実験スクリプトと、LLMが編集プログラムを書いて試行を重ねる試行ループがあります。LLMを呼ぶ実験には `ANTHROPIC_API_KEY` が要ります。`.env` に置き、`uv run --env-file .env python ...` で実行してください（`.env` は git で追跡しません）。

## 使い方

編集（1つの重み行列への一回きりの書き換え）：

```python
import numpy as np
import weightir as wir

W = np.random.randn(4, 6)                          # (出力次元, 入力次元)
K = wir.Keys(np.random.randn(6, 2), "layer")      # 入力側のキー（1列が1例）
V = wir.Values(np.random.randn(4, 2), "layer")    # 出力側の目標値
U = wir.span(np.random.randn(6, 2), "layer")      # 応答を変えない入力の部分空間
patch = wir.compile(wir.edit("layer", map=wir.Exact(K, V), preserve=wir.Exact(U)), W)
patch.delta            # ΔW
patch.check            # 元データでの残差と予算（矛盾した指定なら wir.Infeasible を送出）
```

実モデル（Hugging Face）では `weightir.model.Model` で重みを読み書きし、`weightir.stats`（キー・共分散）、`weightir.targets`（目標値）、`weightir.presets`（ROME・EMMET・MEMIT・AlphaEdit）を使います。

訓練（optimizer の宣言を1行差し替える）：

```python
from weightir.train import Optimizer, Rule

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Optimizer(model, Rule(direction="momentum", metric="covariance", lr=0.05))
```

LLMが書く編集プログラムで学習する（`ProgramOptimizer`）：

```python
from agentic.program import EXAMPLE_PROGRAM, ProgramOptimizer

optimizer = ProgramOptimizer(model, program=EXAMPLE_PROGRAM)   # program: step(ctx) を定義する Python のソース
...
def closure():
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    return loss
loss = optimizer.step(closure)
optimizer.observe(loss.item(), val_acc=...)   # 検証の成績などは、測ったステップでだけ渡す（プログラムが読める）
```

プログラムは毎ステップ実行され、`ctx` から今のバッチの層の入力（キー）・出力・出力に対する勾配・入力の共分散を受け取り、`ctx.apply(wir.edit(...))` で重みを書き換えます。重みの変化はすべてこの編集から来ます（バイアスは `[W | b]` として一緒に編集）。`ctx.refresh()` で編集の後の損失を測り直し、`ctx.undo(patch)` で取り消せます。

プログラムを書くのは LLM です（`agentic.loop.improve`）。コーディングエージェントと同じく、LLM がプログラムを書き、ハーネスが試行として新しい初期値から1本学習させて報告（検証の成績やリターンの推移、時間、編集の統計、プログラムの `print`）を返し、LLM が直す、を決めた回数だけ繰り返して、最良のプログラムを残します。テストの成績は LLM に見せません。

`Rule` は層ごとの更新を `ΔW = argmin ⟨D, ΔW⟩ + tr(ΔW M ΔWᵀ)/(2·lr)  s.t. ΔW U = 0, ‖ΔW‖_M ≤ max_norm, rank(ΔW) ≤ rank` として決めます。D は勾配から作る方向（sgd・momentum・adam・sign・orth）、M は計量（単位行列か入力の共分散）、U は保存する部分空間です。

## 実験

```sh
# 分類（MNIST / Fashion-MNIST の MLP）
.venv/bin/python experiments/classification/train.py --dataset mnist --optimizer wir-cov --epochs 3 --seed 0
# 強化学習（CartPole-v1、価値関数をベースラインにした REINFORCE）
.venv/bin/python experiments/rl/cartpole.py --optimizer wir-orth --episodes 600 --seed 0
# LLM が編集プログラムを20回試して最良を選び、評価用のシードで学習する（.env の ANTHROPIC_API_KEY を使う）
uv run --env-file .env python experiments/search.py --task mnist --method llm --trials 20
# 比較用：Adam の学習率を同じ20回で選ぶ
.venv/bin/python experiments/search.py --task mnist --method adam-lr --trials 20
# 知識編集（CounterFact でプリセットを評価）
.venv/bin/python experiments/editing/presets.py --model gpt2-xl --method rome --dataset counterfact --size 100
```

最適化器の一覧は [experiments/optimizers.py](experiments/optimizers.py) にあり、各スクリプトは `optimizer = OPTIMIZERS[args.optimizer](model)` の1行で切り替えます。`agent-fixed` は例の編集プログラム（LLMなし）です。LLM が試行で選んだプログラム（`agent`）と、学習率を同じ回数で選んだ Adam（`adam-tuned`）は [experiments/search.py](experiments/search.py) が作ります。結果は `data/runs/` に保存されます。実験の定義と結果は [docs/experiments.md](docs/experiments.md) にあります。GPU サーバーでは CLAUDE.md の指示どおり `CUDA_VISIBLE_DEVICES=2,3 taskset -c 16-31,48-63` を付けて実行してください。

## セットアップ

```sh
uv sync                          # Python 3.13 の環境を .venv に作る
uv run pytest -q                 # テスト（data/hf の極小モデルを使う）
git submodule update --init      # 参照実装（third_party/、浅い clone）
```

モデル・データセット・実験結果は `data/` に置き、git では追跡しません。論文のPDF（`third_party/papers/`）も追跡せず、[論文一覧](third_party/papers/README.md) のリンクから取得します。

## 構成

```text
agentic_optimization/
├── src/weightir/
│   ├── spec.py        編集の DSL（Keys・Values・Subspace・Covariance、Exact・Soft、Rank・Norm、edit・joint）
│   ├── solve.py       閉形式の求解、予算、矛盾の判定（Infeasible）
│   ├── check.py       元データでの制約の測り直し（bfloat16 などで保存した後も）
│   ├── compile.py     compile と then
│   ├── train.py       訓練用の Rule と Optimizer
│   ├── model.py       Hugging Face モデルの重みの読み書き
│   ├── nethook.py     フック
│   ├── stats.py       キー、2次モーメント、部分空間
│   ├── targets.py     目標値の最適化（ROME / MEMIT）
│   └── presets.py     ROME・EMMET・MEMIT・AlphaEdit
├── src/agentic/
│   ├── llm.py         Claude の呼び出し
│   ├── program.py     ProgramOptimizer（編集プログラムを毎ステップ動かす）
│   └── loop.py        試行ループ（LLM がプログラムを書き、試行の結果を見て直す）
├── evals/             CounterFact の指標、コーパスと統計量のキャッシュ
├── experiments/       分類・強化学習・編集の実験スクリプト
├── tests/             テスト
├── docs/              設計、実験、関連研究
└── third_party/       参照実装（submodule）と論文の一覧
```

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 設計 |
| [docs/experiments.md](docs/experiments.md) | 実験の定義と結果、LLM の費用 |
| [docs/related_works.md](docs/related_works.md) | 関連研究：7分野・141件と、構想への含意 |
| [third_party/README.md](third_party/README.md) | 参照実装（submodule 35件）の一覧 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 移植したコードの著作権表示とライセンス |
