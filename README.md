# agentic_optimization

LLMに重みの数値を書かせる代わりに、**型と契約を持つ重み編集プログラム**を書かせる研究のリポジトリです。コンパイラはプログラムを制約つき最適化問題に変換して解き、独立に検証したうえで、次の2つを出力します。

- 重みの差分（パッチ）
- 何が、どの前提で保証されているかの記録（マニフェスト）

> 現状：設計の段階です。コードはまだありません。

## 考え方

重みは、関数の実装の一つにすぎません。隠れユニットの並べ替えや残差ストリームの回転によって、同じ関数を実装する重みはいくらでも作れます。そこで、編集を「何番目の値をいくつ変えるか」ではなく、線形代数の言葉で書きます。

```text
最小化  tr(ΔW C ΔWᵀ)       変化量（計量 C は明示する）
制約    (W + ΔW) K = V      キー K に値 V を返す
        ΔW U = 0            部分空間 S（基底 U）への応答は変えない
予算    rank(ΔW) ≤ 8, ‖ΔW‖ ≤ B
```

LLMは「どの層の、どのキーを、何を保存して変えるか」を書き、数値はコンパイラが解きます。ROME・MEMIT・EMMET・AlphaEdit は、この形の仕様の具体例として表せます。同じ制約は、学習の更新則（毎ステップ実行する最適化プログラム）にも変換します。

この研究の主な問いは次の3つです。

1. 由来（どのモデル状態から作った統計量か）と合成則を型で扱うと、編集の失敗を事前に検出できるか。
2. 保証を「代数的・数値的・統計的・行動評価」に分けて記録すると、主張の取り違えを防げるか。
3. 同じ数値計算部分を使ったとき、LLMは素のPythonより、この言語で良い編集を書けるか。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 設計：型、コンパイルの流れ、ソルバ、合成の意味論、マニフェスト、最適化器への変換、評価、マイルストーン |
| [docs/related_works.md](docs/related_works.md) | 関連研究：7分野・141件と、それぞれの構想への含意 |
| [third_party/README.md](third_party/README.md) | 参照実装（git submodule 35件）の一覧と使い道 |
| [third_party/papers/README.md](third_party/papers/README.md) | 論文の一覧（PDFはgitで追跡しない） |

## セットアップ

```sh
git clone --recurse-submodules <このリポジトリのURL>
# すでに clone してある場合
git submodule update --init
```

submodule はすべて浅い clone です。論文のPDF（約470MB）は再配布できないものが多いので、gitでは追跡せず、`third_party/papers/` に手元でだけ置いています。別の環境では、[論文一覧](third_party/papers/README.md) のリンクから取得してください。

## リポジトリ構成

```text
agentic_optimization/
├── README.md
├── docs/
│   ├── architecture.md        設計
│   └── related_works.md       関連研究
└── third_party/
    ├── README.md              submodule の一覧
    ├── papers/                論文（一覧の README.md だけを追跡し、PDFは追跡しない）
    └── <submodule>/ ×35       参照実装と評価データ
```

今後、`src/weightir/`（言語・コンパイラ・検証器）、`src/agentic/`（LLMプランナー）、`evals/`、`experiments/`、`tests/` を追加する予定です。詳しくは [architecture.md の13節](docs/architecture.md#13-リポジトリ構成予定) を参照してください。

## ロードマップ

| 段階 | 内容 |
|---|---|
| M0 | 数学カーネルと性質テスト |
| M1 | 実モデルでの単一サイト編集。ROME・MEMIT・EMMET・AlphaEdit をプリセットとして再現 |
| M2 | 合成（joint / then / merge）、陳腐化の検出、逐次編集 |
| M3 | LLMエージェントと比較実験（Python・JSON・型つきDSL） |
| M4 | 正解の部分空間が分かるテストベッド（Tracr、InterpBench） |
| M5 | 最適化プログラムへの変換（制約つきファインチューニング） |
| M6 | 出力レベルの仕様と統計的保証 |
