# 関連研究

この構想（LLMが、型と契約を持つ重み編集プログラムを書き、コンパイラが線形代数でパッチに変換する）に関わる研究を、7分野に分けてまとめます。設計への反映は [architecture.md](architecture.md) を参照してください。

- 各項目の「PDF」は `third_party/papers/` の中を指します。PDFはgitで追跡していないので、別の環境では論文名のリンクから取得してください。一覧表は [third_party/papers/README.md](../third_party/papers/README.md) にあります。
- 主要な論文は本文まで読んで確認しました。発表先は、予稿集・OpenReview・arXivのjournal-refで確認しています。2026年のプレプリントは査読前です。
- 「→」の後は、この構想にとっての意味です。T0〜T4 は [experiments.md](experiments.md) のテストベッドの記号です。

## 0. 要約

### 既にあるもの（単独では新規性にならない）

- 保存部分空間つきの閉形式の重み編集（ROME、MEMIT、EMMET、AlphaEdit）
- 重みを操作する宣言的な言語（mergekit、BrainSurgery）
- 「仕様 → 最小変更 → 検証」という契約（PRDNN、APRNN、PREPARED、Vehicle）
- LLMや探索がアルゴリズムをコードとして書き、評価で選ぶ仕組み。型検査を含む（Lion、AlphaEvolve、DiscoPOP、OPTScientist）
- 保存部分空間を学習の更新則に組み込む方法（GPM、Adam-NSCL、InfLoRA、NSPO）

### 見つからなかったもの（新規性の候補）

1. joint・then・merge の合成則と、それが壊れる条件（多層、ソフト制約、圧縮）を明示した、編集の代数
2. 同じ数値計算部分を使ったとき、LLMが素のPythonより型つきの言語で良い編集を書けるかの実証

### 既存の閉形式編集は、同じ仕様の仲間

| 手法 | 編集 | 保存 | 最小化する量 |
|---|---|---|---|
| WeightIR の編集 | 厳密またはソフト | 部分空間を厳密に、または C でソフトに | tr(ΔW M ΔWᵀ)（M = C、または M = I で ‖ΔW‖_F²） |
| ROME | 厳密・1件 | 汎用テキストのキー共分散Cでソフトに | E‖ΔW k‖² = tr(ΔW C ΔWᵀ) |
| MEMIT | ソフト（最小二乗） | λC でソフトに | ‖ΔWK − R‖² + λ tr(ΔW C ΔWᵀ) |
| EMMET | 厳密・バッチ | Cでソフトに | tr(ΔW C ΔWᵀ)（実装では αI を足すので厳密性が緩む） |
| AlphaEdit | ソフト | 閾値つきの零空間（固有値1e-2以下）＋リッジ | 編集誤差＋正則化 |
| ENCORE | ソフト | λ_p C0 ＋ Frobenius 項 | ‖ΔWK − R‖² + λ_p tr(ΔW C0 ΔWᵀ) + λ_n‖ΔW‖² |
| MALMEN | ソフト | なし（リッジのみ） | ‖SU − D‖² + λ‖S‖² |
| UCE（画像生成） | ソフト | 名前つきの保存リストでソフトに | 最小二乗 |

ROME の更新式は、「新しいキーと値の対応を厳密に満たす中で、C計量での変化量を最小にする解」と一致します（数値でも確認済み）。C計量はどんな可逆な座標変換でも解が変わりませんが、‖ΔW‖_F は直交変換以外では変わります。

---

## 1. 閉形式の重み編集と、その評価（editing）

### 1.1 手法

- **[Editable Neural Networks](https://arxiv.org/abs/2004.00345)**（Sinitsin et al., ICLR 2020）[PDF](../third_party/papers/editing/sinitsin2020_editable_nn.pdf)
  編集を「ある制約を満たし、対照集合での振る舞いは保つ」問題として最初に定式化した。メタ学習で、勾配による編集が成功しやすいモデルを作る。検査は経験的なものだけで、実現できない仕様を検出する仕組みはない。
  → 契約の形（何を変え、何を保つか）の原型。

- **[Rewriting a Deep Generative Model](https://arxiv.org/abs/2007.15646)**（Bau et al., ECCV 2020）[PDF](../third_party/papers/editing/bau2020_rewriting.pdf)
  層を線形の連想記憶とみなし、`Wk* = v*` を満たしつつ既存のキーと値の対応をなるべく崩さないランク1更新を解く。ユーザーの「コピー／ペースト／文脈」の操作を、値・キー・更新部分空間（白色化したPCA）に変換する。
  → 構想の流れ（操作 → K・V・S → 閉形式）がすでにある。ただし型も制約の検査もなく、対象は1層だけ。

- **[MEND: Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309)**（Mitchell et al., ICLR 2022）[PDF](../third_party/papers/editing/mitchell2022_mend.pdf)
  ランク1に分解した勾配を、ハイパーネットワークで編集用の差分に変換する。100億パラメータ超まで使える。著者自身、ランダムな負例では難しさが足りず、編集が広がりすぎると述べている。
  → 学習済みの「編集コンパイラ」。比較対象にする。

- **[ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)**（Meng et al., NeurIPS 2022）[PDF](../third_party/papers/editing/meng2022_rome.pdf)
  `ΔW = (v* − Wk*)(C⁻¹k*)ᵀ / (k*ᵀC⁻¹k*)`。C は Wikipedia の10万サンプルから推定したキーの（中心化しない）2次モーメント。k* は主語の最後のトークンでの平均キー、v* は勾配法で最適化する。
  → `Exact(K, V)` が1件で `minimize=C`、保存なしの編集にあたる（`presets.rome`）。C は宣言した保存集合ではなく、汎用テキストの統計。

- **[MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)**（Meng et al., ICLR 2023）[PDF](../third_party/papers/editing/meng2023_memit.pdf)
  `Δ = R K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹`、`C₀ = λE[kkᵀ]`。残差を複数の層に分け、層ごとに、途中まで編集したモデルでキーを計算し直す。1万件の編集で評価。
  → 各層で `Soft(K, V)` と `minimize=λC`、保存なしの編集にあたる（`presets.memit`）。層ごとにキーを計算し直すのは、`then` が後の編集を編集後のモデルで組み立てるのと同じ考え方。

- **[MALMEN: Massive Editing for LLMs via Meta Learning](https://arxiv.org/abs/2311.04661)**（Tan et al., ICLR 2024）[PDF](../third_party/papers/editing/tan2024_malmen.pdf)
  ハイパーネットワークが出したトークンごとの差分を、単純に足さずに `S* = DUᵀ(UUᵀ + λI)⁻¹` で最小二乗にまとめる。保存項はなく、局所性はメタ学習に任せる。
  → 「学習で目標を作り、最小二乗でまとめる」設計の先例。比較対象にする。

- **[Unified Concept Editing in Diffusion Models](https://arxiv.org/abs/2308.14761)**（Gandikota et al., WACV 2024）[PDF](../third_party/papers/editing/gandikota2024_uce.pdf)
  画像生成モデルのクロスアテンションを、編集リストと名前つきの保存リストを使った閉形式で書き換える。100人を超える画家を消すと、保存したはずの画家にも影響が出始める。
  → 同じ「編集＋保存」の型が別のモダリティでも成り立つ。保存リストを名前で明示する点が参考になる。

- **[A Unified Framework for Model Editing](https://arxiv.org/abs/2403.14236)**（Gupta et al., EMNLP 2024 Findings）[PDF](../third_party/papers/editing/gupta2024_emmet.pdf)
  ROME と MEMIT を「保存と記憶」の目的関数でまとめ、等式制約つきのバッチ編集 EMMET（`Δ = R(K_EᵀC₀⁻¹K_E)⁻¹K_EᵀC₀⁻¹`）を導く。条件数が悪いので `+αI` を足しており、実質的に等式を緩めている。EMMET と MEMIT は下流タスクを同じように劣化させる。
  → 「厳密な編集」は保存を意味しない。実装が黙って厳密性を緩める例でもある。`presets.emmet` は、論文に書かれた厳密な目的関数を解く。

- **[Rebuilding ROME](https://arxiv.org/abs/2403.07175)**（Gupta et al., EMNLP 2024）[PDF](../third_party/papers/editing/gupta2024_rebuilding_rome.pdf)
  ROME の逐次編集でモデルが壊れる原因は、元の実装が2種類のキー（平均キーと生のプロンプトのキー）を混ぜていたことだった。直した r-ROME では崩壊しない。
  → 実装の不整合が「手法の限界」に見えることがある。独立した検証器が要る理由の一つ。

- **[AlphaEdit: Null-Space Constrained Knowledge Editing](https://arxiv.org/abs/2410.02355)**（Fang et al., ICLR 2025, Outstanding Paper）[PDF](../third_party/papers/editing/fang2025_alphaedit.pdf)
  保存するキーの `K₀K₀ᵀ` の固有ベクトルのうち、固有値が1e-2を超えるものを取り除き、残りで射影 P を作る（公開コードでは平均した2次モーメントに対して 2e-2）。最終式は `Δ = RK₁ᵀP(K_pK_pᵀP + K₁K₁ᵀP + I)⁻¹`。P は編集前のモデルで一度だけ計算する。
  → 「零空間に制限する」は閾値つきの近似で、厳密な保存ではない。WeightIR では推定した部分空間（`exact=False`）として厳密な部分空間と区別し、`check` が元データでの出力の変化を報告する。`presets.alphaedit` は公開コードの2e-2を使う。

- **[Lifelong Knowledge Editing requires Better Regularization (ENCORE)](https://arxiv.org/abs/2502.01636)**（Gupta et al., EMNLP 2025 Findings）[PDF](../third_party/papers/editing/gupta2025_encore.pdf)
  劣化の原因を、活性値の過剰な最適化と重みのノルムの増大に求める。`RK₁ᵀ(λ_pC₀ + K₁K₁ᵀ + λ_nI)⁻¹` で1万件の逐次編集に耐える。AlphaEdit はリッジ項を外すと崩壊すると報告している。
  → ノルムの予算（`Norm(B)`）が実際に効く根拠。

- **[Rethinking Residual Distribution in Locate-then-Edit Model Editing (BLUE)](https://arxiv.org/abs/2502.03748)**（Li et al., NeurIPS 2025）[PDF](../third_party/papers/editing/li2025_blue.pdf)
  MEMIT 型の「残差を複数層に分配する」操作そのものが誤差を生むことを示し、境界の層だけを編集する。
  → 複数サイトの編集は、層ごとの解を足すだけでは済まない。

- **[FisherEdit](https://arxiv.org/abs/2610.25914)**（Yang et al., 2026 プレプリント）[PDF](../third_party/papers/editing/yang2026_fisheredit.pdf)
  AlphaEdit の射影 P はほぼ恒等写像で（閾値を超える方向は数十しかない）、外しても結果がほとんど変わらないと報告。代わりに対角 Fisher による要素ごとのペナルティを使う。
  → 対角 Fisher の計量は座標に依存する（C計量は依存しない）。

- **[BetaEdit: Null-Space Constrained Sequential Model Editing](https://www.ijcai.org/proceedings/2026/0508.pdf)**（Liu et al., IJCAI 2026）[PDF](../third_party/papers/editing/liu2026_betaedit.pdf)
  「近似零空間」は知識を漏らし、逐次編集を劣化させると示す。
  → 閾値で推定した部分空間を厳密な部分空間と型で区別し（`exact=False`）、元データを持たせる理由。

### 1.2 層についての保証と、モデル全体の振る舞い

- **[Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213)**（Hase et al., NeurIPS 2023）[PDF](../third_party/papers/editing/hase2023_localization.pdf)
  ROME の編集の成否は、ほぼ「どの層を選ぶか」で決まり、因果トレーシングで得た局在の結果はほとんど説明力を持たない。
  → 局在は候補を出すのに使い、成否は検証で判断する。

- **[MQuAKE](https://arxiv.org/abs/2305.14795)**（Zhong et al., EMNLP 2023）[PDF](../third_party/papers/editing/zhong2023_mquake.pdf)
  編集した事実は答えられても、それを使う多段の質問には大きく失敗する。
  → 層についての保証は、推論の連鎖には及ばない。評価（T3）に使う。

- **[Evaluating the Ripple Effects of Knowledge Editing (RippleEdits)](https://arxiv.org/abs/2307.12976)**（Cohen et al., TACL 2024）[PDF](../third_party/papers/editing/cohen2024_rippleedits.pdf)
  5千件の編集と6つの基準で、編集から論理的に導かれる事実への波及を測る。編集手法はどれも波及に失敗し、文脈内で知識を与える単純な方法が最も良い。
  → 評価（T3）に使う。

- **[Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://arxiv.org/abs/2401.07453)**（Gupta et al., ACL 2024 Findings）[PDF](../third_party/papers/editing/gupta2024_forgetting.pdf)
  ROME と MEMIT の逐次編集で、忘却が徐々に、やがて急激に進む。モデルを壊す編集は、ノルムが大きい。

- **[Model Editing Harms General Abilities of LLMs](https://arxiv.org/abs/2401.04700)**（Gu et al., EMNLP 2024）[PDF](../third_party/papers/editing/gu2024_editing_harms.pdf)
  4手法・3モデル・8タスクで、汎用能力の劣化を示す。対策として、更新のうち相対的に大きい成分だけを残す（RECT）。

- **[The Butterfly Effect of Model Editing](https://arxiv.org/abs/2402.09656)**（Yang et al., ACL 2024 Findings）[PDF](../third_party/papers/editing/yang2024_butterfly.pdf)
  1回の編集でモデルが崩壊することがある。安価な指標としてパープレキシティを提案する。

- **[Knowledge in Superposition](https://arxiv.org/abs/2408.07413)**（Hu et al., AAAI 2025）[PDF](../third_party/papers/editing/hu2025_knowledge_superposition.pdf)
  逐次編集で干渉が起きないには、キー同士が白色化した空間で直交している必要がある。調べたすべてのモデルと層で、この直交性は破れていた。先に入れた編集ほど干渉を受ける。
  → `merge` を例外扱いにし、`then` で過去の編集を保存する設計の根拠。

- **[The Mirage of Model Editing](https://arxiv.org/abs/2502.11177)**（Yang et al., ACL 2025）[PDF](../third_party/papers/editing/yang2025_mirage.pdf)
  teacher forcing による評価が成功率を水増ししていた。実際の生成で測ると、報告値の96.8%が38.5%に下がる。
  → 評価は実際の生成で行う。

- **[Reproducibility Study of AlphaEdit](https://arxiv.org/abs/2606.26783)**（Suresh et al., 2026 プレプリント）[PDF](../third_party/papers/editing/suresh2026_alphaedit_reproducibility.pdf)
  約5千件を超えると保護が弱まり、Phi-3 と Gemma-2 では機能しなかった。Gemma-2 の FFN 後の RMSNorm が、導出の前提（更新が残差ストリームに線形に入る）を崩している可能性がある。
  → 保存の主張は「一つの層、編集前のキー」という前提つきで書く。正規化層の位置によっては、この前提が崩れる。

### 構想への含意

- 既存手法は「仕様のプリセット」として表せる。新規性はソルバではなく、仕様・型・合成・検証の側にある。
- 層についての厳密な式は、モデル全体の振る舞いを保証しない。この差を測る評価（波及、多段推論、逐次編集、汎用能力）が必須。
- 実装の不整合や黙った緩和（ROME のキーの混在、EMMET の αI）があるので、元データでの独立検証が必要。

---

## 2. プログラム ⇄ Transformer の重み（compilers）

- **[Thinking Like Transformers (RASP)](https://arxiv.org/abs/2106.06981)**（Weiss et al., ICML 2021）[PDF](../third_party/papers/compilers/weiss2021_rasp.pdf)
  Transformer エンコーダの計算を、select・aggregate・selector_width と要素ごとの演算で書く言語。論文の「コンパイラ」は層とヘッドの数を出すところまでで、具体的な重みは対象外。

- **[Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062)**（Lindner et al., NeurIPS 2023）[PDF](../third_party/papers/compilers/lindner2023_tracr.pdf)
  RASP をデコーダ型 Transformer の重みに変換する。論文自身が「プログラミング言語をアセンブリ、機械語へと翻訳するのと同じ」とたとえ、中間表現 Craft を「アセンブリ言語」と呼ぶ。Craft は名前つきの基底方向を持つベクトル空間とその間の写像で、各変数は専用の直交部分空間を持つ。有限の語彙と最大長から値域を推論し、数値のMLPやソフトマックスでは近似を使い、LayerNorm は取り除く。著者は「完全な言語モデルを Tracr でコンパイルすることはおそらくない」と述べ、学習済みモデルの一部を置き換える用途を今後の課題に挙げている。
  → 「言語 → 線形代数の中間表現 → 重み」の先例。ただし座標軸に沿った直交部分空間だけで、重ね合わせ・制約・最適化はない。チェックポイントの「名前つき空間」は推定して作るしかない。

- **[Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196)**（Giannou et al., ICML 2023）[PDF](../third_party/papers/compilers/giannou2023_looped.pdf)
  手で重みを設定したループ型の13層 Transformer が、汎用計算機として電卓や線形代数を実行する。手で作ったモデルを大きな学習済みモデルと融合することを、未解決問題として挙げる。

- **[Learning Transformer Programs](https://arxiv.org/abs/2306.01128)**（Friedman et al., NeurIPS 2023）[PDF](../third_party/papers/compilers/friedman2023_transformer_programs.pdf)
  離散的なプログラムに変換できるよう制約した Transformer を学習し、Python に変換する。入力が長く語彙が大きいと、性能が大きく落ちる。

- **[What Algorithms can Transformers Learn? (RASP-L)](https://arxiv.org/abs/2310.16028)**（Zhou et al., ICLR 2024）[PDF](../third_party/papers/compilers/zhou2024_rasp_l.pdf)
  短い RASP-L プログラムがあるタスクでは長さへの汎化が起きる、という予想。学習した重みがコンパイルした重みに近いとは主張しない。

- **[Mastering Symbolic Operations: Augmenting LMs with Compiled Neural Networks](https://arxiv.org/abs/2304.01665)**（Weng et al., ICLR 2024）[PDF](../third_party/papers/compilers/weng2024_mastering_symbolic_operations.pdf)
  Tracr でコンパイルしたモジュールを学習済みLMの**横に**置き、出力語彙上のゲートで切り替える。LMの重みは書き換えない。GPT-3.5 に Tracr プログラムを書かせる AutoCoNN は、足し算と引き算で失敗した。

- **[InterpBench](https://arxiv.org/abs/2407.14494)**（Gupta et al., NeurIPS 2024 Datasets & Benchmarks）[PDF](../third_party/papers/compilers/gupta2024_interpbench.pdf)
  Tracr の回路を、現実的な重みを持つモデルとして学習し直した86個のモデル。どの回路を実装しているかが分かっている。
  → 正解の部分空間が分かるテストベッド（T1）。

- **[TracrBench](https://arxiv.org/abs/2409.13714)**（Thurnherr & Scheurer, ICML 2024 MI Workshop）[PDF](../third_party/papers/compilers/thurnherr2024_tracrbench.pdf)
  GPT-4-turbo が正しく書けた RASP は101問中57問。同じアルゴリズムを Python で書くと約96%が正解。
  → DSLをPythonに埋め込む理由。

- **[Neural Decompiling of Tracr Transformers](https://arxiv.org/abs/2410.00061)**（Thurnherr & Riesen, ANNPR 2024）[PDF](../third_party/papers/compilers/thurnherr2024_neural_decompiling.pdf)
  重みから RASP プログラムを復元する。完全に一致するのは3割強、機能が同じものは7割強。
  → 同じ機能を実装する重み（とプログラム）は一つではない。

- **[ALTA: Compiler-Based Analysis of Transformers](https://arxiv.org/abs/2410.18077)**（Shaw et al., TMLR 2025）[PDF](../third_party/papers/compilers/shaw2025_alta.pdf)
  Tracr にループと Universal Transformer を加えた。プログラムの層ごとの出力を教師にして MLP だけを学習させる方法（trace supervision）で、学習しやすさを分析する。
  → 「仕様 → 層ごとの入出力の目標 → 層ごとに当てはめる」の先例。ただし白紙から学習し、保存の制約はない。

- **[Tracr-Injection](https://arxiv.org/abs/2505.10719)**（Vergara-Browne & Soto, ACL 2025 Findings）[PDF](../third_party/papers/compilers/vergarabrowne2025_tracr_injection.pdf)
  GPT-2-large の残差ストリームの線形写像が Tracr の活性値に合うよう、ファインチューニングで蒸留する。保証はなく、保存も経験的にしか示されない。
  → 既存LLMにコンパイル済みの計算を入れる、最も近い試み。

- **[Discovering Interpretable Algorithms by Decompiling Transformers to RASP](https://arxiv.org/abs/2602.08857)**（Huang et al., ICML 2026）[PDF](../third_party/papers/compilers/huang2026_decompiling_to_rasp.pdf)
  小さな GPT-2 型のモデルを、LayerNorm を線形とみなす仮定のもとで D-RASP プログラムに書き直し、因果的に刈り込む。

- **[Synthesis and Verification of Transformer Programs](https://arxiv.org/abs/2602.16473)**（Jiang et al., 2026 技術報告）[PDF](../third_party/papers/compilers/jiang2026_transformer_program_verification.pdf)
  C-RASP プログラムをモデル検査（Lustre / SMT）で検証し、例や部分的な仕様から局所探索で合成する。

### 構想への含意

- 「言語 → 線形代数の中間表現 → 重み」は、白紙のモデルでは実現している。既存LLMの層に、保存制約と保証つきでコンパイルしたパッチを入れた研究は見つからなかった。
- どの研究も LayerNorm を取り除くか線形とみなしている。実際のLLMには正規化層があるので、層についての保証がどこまで及ぶかは、正規化の位置に左右される。

---

## 3. 重みの代数・マージ・対称性・重み操作言語（merging）

### 3.1 足し算が効く条件

- **[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)**（Ilharco et al., ICLR 2023）[PDF](../third_party/papers/merging/ilharco2023_task_arithmetic.pdf)
  `τ = θ_ft − θ_pre` を足し引きして、タスクの追加・忘却・類推を行う。干渉が小さい理由（ほぼ直交）は推測にとどまる。

- **[Task Arithmetic in the Tangent Space](https://arxiv.org/abs/2305.12827)**（Ortiz-Jimenez et al., NeurIPS 2023）[PDF](../third_party/papers/merging/ortizjimenez2023_tangent_space.pdf)
  「重みの分離可能性」を定義する。各タスクの入力の担当範囲が重ならないとき、タスクベクトルの和は、各範囲でそれぞれのタスクベクトルだけを足した場合と同じに振る舞う。線形化したファインチューニングで分離可能性が上がる。
  → 合成則の最も形式的な先例。ただし、同じ入力に作用する更新同士の合成は扱わない。

- **[When is Task Vector Provably Effective for Model Editing?](https://arxiv.org/abs/2504.10957)**（Li et al., ICLR 2025）[PDF](../third_party/papers/merging/li2025_task_vector_theory.pdf)
  1層の Transformer で、タスクの足し算が効く十分条件を示す。矛盾するタスクでは適切な係数を見つけられない。

- **[TIES-Merging](https://arxiv.org/abs/2306.01708)**（Yadav et al., NeurIPS 2023）[PDF](../third_party/papers/merging/yadav2023_ties.pdf) と **[DARE](https://arxiv.org/abs/2311.03099)**（Yu et al., ICML 2024）[PDF](../third_party/papers/merging/yu2024_dare.pdf)
  要素ごとの経験則で干渉を減らす（上位k%を残して符号で投票する／ランダムに落として拡大する）。理論的な裏付けは弱い。

- **[Task Singular Vectors](https://arxiv.org/abs/2412.00081)**（Gargiulo et al., CVPR 2025）[PDF](../third_party/papers/merging/gargiulo2025_task_singular_vectors.pdf)
  層ごとのSVDで干渉を定義し、直交化してから統合する。

- **[Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832)**（Matena & Raffel, NeurIPS 2022）[PDF](../third_party/papers/merging/matena2022_fisher_merging.pdf)
  対角 Fisher で重みづけた平均。単純平均は等方的な特別な場合。

- **[RegMean: Dataless Knowledge Fusion by Merging Weights](https://arxiv.org/abs/2212.09849)**（Jin et al., ICLR 2023）[PDF](../third_party/papers/merging/jin2023_regmean.pdf)
  線形層ごとに、出力の差の二乗和を最小にする閉形式 `W = (Σ W_i K_i K_iᵀ)(Σ K_i K_iᵀ)⁻¹`（この文書の列の向きで書いた形）。
  → `merge` を作るときの数学的な土台。出力空間で定義するので、入力の座標変換で解が変わらない。

- **[Merging by Matching Models in Task Parameter Subspaces (MaTS)](https://arxiv.org/abs/2312.04339)**（Tam et al., TMLR 2024）[PDF](../third_party/papers/merging/tam2024_mats.pdf)
  マージを線形方程式 `θ* = (ΣC_m)⁻¹ΣC_mθ_m` とみなして共役勾配法で解く。平均・Fisher・RegMean を同じ形で扱える。

### 3.2 対称性とゲージ

- **[Git Re-Basin](https://arxiv.org/abs/2209.04836)**（Ainsworth et al., ICLR 2023）[PDF](../third_party/papers/merging/ainsworth2023_git_rebasin.pdf)
  隠れユニットの並べ替えで2つのモデルを揃えてから統合する。並べ替え以外の対称性（拡大縮小など）の役割は未解明としている。

- **[SliceGPT](https://arxiv.org/abs/2401.15024)**（Ashkboos et al., ICLR 2024）[PDF](../third_party/papers/merging/ashkboos2024_slicegpt.pdf)
  RMSNorm でつながった Transformer では、残差ストリームを任意の直交行列で回転しても関数が変わらない（定理1）。LayerNorm のモデルは先に変換が要る。
  → 残差ストリーム側では ‖ΔW‖_F が座標に依存しない根拠。

- **[QuaRot](https://arxiv.org/abs/2404.00456)**（Ashkboos et al., NeurIPS 2024）[PDF](../third_party/papers/merging/ashkboos2024_quarot.pdf)
  同じ不変性を使い、アダマール回転を重みに埋め込んで量子化する。
  → 同じ関数が別の座標で配布されうる。座標に依存する計量（‖ΔW‖_F）の解は、配布された座標によって変わる。

- **[Equivariant Neural Functional Networks for Transformers](https://arxiv.org/abs/2410.04209)**（Tran et al., ICLR 2025）[PDF](../third_party/papers/merging/tran2025_attention_symmetry.pdf)
  多頭注意の対称性の最大の群が、ヘッドの並べ替えとヘッドごとの一般線形変換 GL(D_k)、GL(D_v) であることを示す。
  → W_Q や W_K を単独で ‖ΔW‖_F 最小化すると座標依存になる。積を対象にするか、ゲージを固定する。

- **重みを入力とするネットワーク**：[DWSNets](https://arxiv.org/abs/2301.12780)（Navon et al., ICML 2023）[PDF](../third_party/papers/merging/navon2023_dwsnets.pdf)、[NFN](https://arxiv.org/abs/2302.14040)（Zhou et al., NeurIPS 2023）[PDF](../third_party/papers/merging/zhou2023_nfn.pdf)、[Graph Metanetworks](https://arxiv.org/abs/2312.04501)（Lim et al., ICLR 2024）[PDF](../third_party/papers/merging/lim2024_graph_metanetworks.pdf)
  並べ替えに対して同変なネットワークで重みを処理する。編集言語ではないが、重み空間の対称性の扱い方の参考になる。

### 3.3 重みを操作する言語と、その探索

- **[Arcee's MergeKit](https://arxiv.org/abs/2403.13257)**（Goddard et al., EMNLP 2024 Industry）[PDF](../third_party/papers/merging/goddard2024_mergekit.pdf)
  YAMLでマージ手順（手法、層の範囲、パラメータ）を書く。手順を有向グラフにつなぐことや、CMA-ES での探索もできる。検査するのは名前と形だけで、入出力空間・制約・計量・保証は書けない。

- **[Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187)**（Akiba et al., Nature Machine Intelligence 2025）[PDF](../third_party/papers/merging/akiba2025_evolutionary_merge.pdf)
  パラメータ空間（層ごとの TIES・DARE の係数）とデータフロー空間（モデルをまたぐ層の経路）を、進化計算で探索する。
  → 「重み操作の言語で書いたプログラムを探索する」先例。ただし固定のテンプレートを探索し、評価はベンチマークの点数。

- **[BrainSurgery](https://arxiv.org/abs/2606.09707)**（Barmina et al., 2026 プレプリント）[PDF](../third_party/papers/merging/barmina2026_brainsurgery.pdf)
  重みの copy・slice・scale・matmul・低ランク化を1行ずつYAMLで書くDSL。assert（存在・形・dtype・値）と、実行した手順の記録がある。静的な型、制約の求解、部分空間、保証の記録、LLMによる作成はない。
  → パッチの書き出し先（実行系）として流用できる可能性がある。

- **[LoraHub](https://arxiv.org/abs/2307.13269)**（Huang et al., COLM 2024）[PDF](../third_party/papers/merging/huang2024_lorahub.pdf)
  LoRA モジュールの重みつき和の係数を、少数の例で勾配なしに最適化する。LoRA の因子（A, B）の和は更新の和ではない（交差項が出る）。
  → 演算は因子ではなく更新に対して型づけする。

- **[Can LLMs Invent Algorithms to Improve Themselves? (Self-Developing)](https://arxiv.org/abs/2410.15639)**（Ishibashi et al., NAACL 2025）[PDF](../third_party/papers/merging/ishibashi2025_self_developing.pdf)
  LLM がタスクベクトルを統合する関数を Python で書き、良い関数を書けるよう DPO で学習する。GSM8k で 70.1 → 76.1（Task Arithmetic は 71.9）。
  → 「LLM が重みを操作するプログラムを書く」最も近い先行研究。言語は素の Python で、制約も保証もない。

### 3.4 合成の実証研究

- **[Composable Interventions for Language Models](https://arxiv.org/abs/2407.06483)**（Kolbeinsson et al., ICLR 2025）[PDF](../third_party/papers/merging/kolbeinsson2025_composable_interventions.pdf)
  Llama-3-8B で、知識編集・圧縮・アンラーニングの組み合わせを系統的に調べた。圧縮は編集とアンラーニングを損ない、順序で結果が変わる。編集手法の中では MEMIT が最も合成しやすく、LoRA は合成しにくい。MMLU の点数ではこれらが見えない。
  → 量子化や枝刈りの後は再検証が要る。合成の評価プロトコル（T3）に使う。

- **[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)**（Biderman et al., TMLR 2024）[PDF](../third_party/papers/merging/biderman2024_lora_forgets_less.pdf)
  フルのファインチューニングが学ぶ更新のランクは、典型的な LoRA の設定の10〜100倍。

- **[LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228)**（Shuttleworth et al., NeurIPS 2025）[PDF](../third_party/papers/merging/shuttleworth2025_lora_illusion.pdf)
  LoRA は元の重みにない特異方向（intruder dimensions）を作り、忘却を招く。逐次的なファインチューニングで蓄積する。
  → ランクの予算は容量を制限するだけで、局所性は保証しない。

### 構想への含意

- 単一サイト・厳密制約の合成則は短い証明で書ける（`then` で過去を保存すれば `joint` と同じ、`merge` は干渉項がゼロのときだけ `joint` と同じ）。多層・ソフト制約・非線形・圧縮後の法則は、まだない。
- 重み操作の言語（mergekit、BrainSurgery）は「テンソルをどう組み合わせるか」しか書けない。意味（空間・制約・計量・保証）を持たせるのが、この構想の差分。
- 座標の自由（ゲージ）は残差ストリームでは直交群だが、注意ヘッドや SwiGLU では一般の可逆変換や対角変換になる。計量の選び方は、層の入力空間ごとに決める必要がある。

---

## 4. 仕様・契約・検証つき修復（verification）

- **[DL2: Training and Querying Neural Networks with Logic](https://proceedings.mlr.press/v97/fischer19a.html)**（Fischer et al., ICML 2019）[PDF](../third_party/papers/verification/fischer2019_dl2.pdf)
  論理式を、成り立つときにちょうど0になる損失に変換し、学習と問い合わせに使う宣言的な言語。制約はソフトで、学習後に成り立つ保証はない。

- **[Minimal Modifications of Deep Neural Networks using Verification](https://easychair.org/publications/paper/CWhF)**（Goldberger et al., LPAR-23 2020）[PDF](../third_party/papers/verification/goldberger2020_minimal_modifications.pdf)
  1層の重みの変更を検証器（Marabou）への問い合わせとして解き、二分探索で最小の変更を求める。出力層と ℓ∞ ノルムなら線形計画になる。

- **[Provable Repair of Deep Neural Networks (PRDNN)](https://arxiv.org/abs/2104.04413)**（Sotoudeh & Thakur, PLDI 2021）[PDF](../third_party/papers/verification/sotoudeh2021_prdnn.pdf)
  仕様は、点ごと、または多面体上のすべての点での、出力に対する線形制約。「分離した DNN」にすると出力が1層の重みについて線形になり、ℓ1／ℓ∞ の最小変更を線形計画で求める。解がなければ「その層では修復できない」と返す。どの層を直すかで、性能の低下は1〜5%から40〜59%まで変わる（SqueezeNet）。
  → 「仕様 → 最小変更 → 検証済みの修復か、修復なし」という契約の形がすでにある。

- **[REASSURE](https://arxiv.org/abs/2110.07682)**（Fu & Li, ICLR 2022）[PDF](../third_party/papers/verification/fu2022_reassure.pdf)
  ReLU ネットワークの線形領域ごとに、パッチ用の小さなネットワークを足す（重みではなく関数の側で最小変更）。修復するたびにネットワークが大きくなる。

- **[Architecture-Preserving Provable Repair (APRNN)](https://arxiv.org/abs/2304.03496)**（Tao et al., PLDI 2023）[PDF](../third_party/papers/verification/tao2023_aprnn.pdf)
  アーキテクチャを変えずに修復する。健全で多項式時間だが不完全で、見つかるかは参照点の選び方に依存する。

- **[Provable Editing of DNNs using Parametric Linear Relaxation (PREPARED)](https://par.nsf.gov/biblio/10582138)**（Tao & Thakur, NeurIPS 2024）[PDF](../third_party/papers/verification/tao2024_prepared.pdf)
  入力の多面体上のすべての点で出力が別の多面体に入るように、最小の変更を求める。変える部分のパラメータについて線形な出力の上下界を作り、線形計画で解く。固定した前段には健全な上下界（DeepPoly など）を使うので、モデル全体について保証できる。BERT は最終層だけを編集した。

- **[Provable Repair of Vision Transformers (PRoViT)](https://thakur.cs.ucdavis.edu/assets/pubs/SAIV2024.pdf)**（Nawas et al., SAIV 2024）[PDF](../third_party/papers/verification/nawas2024_provit.pdf)
  Vision Transformer の最終層を、健全だが不完全な線形計画で修復する。2千枚の画像で、性能の低下は0.12%。

- **[Provable Gradient Editing of Deep Neural Networks (ProGrad)](https://proceedings.nips.cc/paper_files/paper/2025/hash/9ecca44ee9150cfe350bcb3660d18c98-Abstract-Conference.html)**（Tao & Thakur, NeurIPS 2025）[PDF](../third_party/papers/verification/tao2025_prograd.pdf)
  入力勾配への硬い制約を線形計画で課す。Llama-3.2-1B と Qwen3-1.7B では LM ヘッドだけを編集する。

- **[WARP: Guaranteed Inner-Layer Repair of NLP Transformers](https://arxiv.org/abs/2604.00938)**（Hsu et al., 2026 プレプリント）[PDF](../third_party/papers/verification/hsu2026_warp.pdf)
  BERT の分類器直前の全結合層を二次計画で修復する。保証は線形化に基づき、修復後に厳密に確かめ直す。

- **[Vehicle](https://arxiv.org/abs/2401.06379)**（Daggitt et al., FSCD 2025）[PDF](../third_party/papers/verification/daggitt2025_vehicle.pdf)
  依存型を持つ仕様言語。一つの仕様を、検証器への問い合わせ・学習の損失関数・Agda の証明に変換する（検証器と損失への変換は Agda で健全性を証明済み）。扱えない仕様は、型検査の説明つきで拒否する。証明はネットワークとデータのファイルのハッシュに紐づき、ファイルが変わると無効になる。
  → 型による拒否の直接の先例。

- **[LEACE: Perfect Linear Concept Erasure in Closed Form](https://arxiv.org/abs/2306.03819)**（Belrose et al., NeurIPS 2023）[PDF](../third_party/papers/verification/belrose2023_leace.pdf)
  どんな線形分類器も概念を定数予測より良く当てられない、という保証つきの消去を閉形式で求める。最小変更は活性値の平均二乗変化で測り、その解はあらゆる内積ノルムについて同時に最適。保証は使ったデータ分布に依存する。
  → 「閉形式＋厳密な保証＋データ依存」の手本。

- **[Refusal in LMs Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)**（Arditi et al., NeurIPS 2024）[PDF](../third_party/papers/verification/arditi2024_refusal_direction.pdf)
  活性値から方向 r̂ を取り除く操作 `x − r̂r̂ᵀx` は、残差ストリームに書き込むすべての行列（埋め込み、注意の出力、MLP の出力、バイアス）を `W − r̂r̂ᵀW` と直交化する重み編集と厳密に等しい。振る舞いへの影響（TruthfulQA の低下）は測定のみ。
  → 活性値への介入を、すべての入力について厳密に重みのパッチへ変換できる例。

- **[Neural Network Verification is a Programming Language Challenge](https://arxiv.org/abs/2501.05867)**（Cordeiro et al., ESOP 2025）[PDF](../third_party/papers/verification/cordeiro2025_verification_pl_challenge.pdf)
  実数上の証明が浮動小数点の実行で破れること、カーネルの自動選択・層の融合・総和の順序が非決定性を生むことを指摘する。
  → 数値の保証は、実際に使う形式で測り直す必要がある。

- **[Provably Safe Model Updates](https://arxiv.org/abs/2512.01899)**（Elmecker-Plakolm et al., SaTML 2026）[PDF](../third_party/papers/verification/elmeckerplakolm2026_provably_safe_updates.pdf)
  パラメータ空間の領域を認証し、その領域に射影した更新はすべて認証済みとする。LLM では、固定した埋め込みの上の線形ヘッドにだけ使っている。
  → 保留データでの統計的な保証の先例。

### 構想への含意

- 契約の形（仕様、最小変更、修復できなければ「なし」）は、PL・形式手法の分野にすでにある。Transformer でモデル全体の保証が得られた例は、最終層か出力層の編集に限られる。
- 出力レベルの修復は ReLU の1層だけでも NP 困難で、実用的な手法の多くは不完全。診断では「不可能と証明した」と「見つからなかった」を分ける。
- 実数上の保証と浮動小数点での保証は別物として測る（`check` は float64 と、保存する dtype の両方で残差を測る）。

---

## 5. 能力の名前と部分空間・内部介入（representations）

### 5.1 理論

- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)**（Elhage et al., 2021, Web記事）
  各部品が残差ストリームの部分空間を読み書きするという見方と、QK回路・OV回路を導入する。残差ストリームには特権的な基底がなく、特権的な基底を持つのはトークン、注意パターン、MLPの活性値だけ。
  → 層の入力空間ごとに基底の性質が違う。計量の選び方の出発点。

- **[The Linear Representation Hypothesis and the Geometry of LLMs](https://arxiv.org/abs/2311.03658)**（Park et al., ICML 2024）[PDF](../third_party/papers/representations/park2024_linear_representation.pdf)
  ソフトマックスの不変性から、ユークリッド内積が自然な選択ではないことを示し、因果的に分離できる概念が直交する「因果内積」を定義する。ただしこの内積は一意ではない（自由度が残る）。ユークリッド内積で概ね足りるモデルもあれば、足りないモデルもある。
  → 計量の選択は飾りではなく、モデルごとに確かめる必要がある。

- **[Causal Abstraction](https://arxiv.org/abs/2301.04709)**（Geiger et al., JMLR 2025）[PDF](../third_party/papers/representations/geiger2025_causal_abstraction.pdf) と **[DAS](https://arxiv.org/abs/2303.02536)**（Geiger et al., CLeaR 2024）[PDF](../third_party/papers/representations/geiger2024_das.pdf)
  高レベルの変数と表現の部分空間の対応を、交換介入で検証する形式的な理論。DAS は、モデルを固定したまま回転を学習し、対応する部分空間を見つける。
  → 「名前 → 部分空間」の意味論の候補。対応づけは最適化で作るもので、検証（交換介入の正答率）とセットになる。

- **[The Non-Linear Representation Dilemma](https://arxiv.org/abs/2507.08802)**（Sutter et al., NeurIPS 2025）[PDF](../third_party/papers/representations/sutter2025_nonlinear_representation_dilemma.pdf)
  非線形な対応づけを許すと、妥当な仮定のもとで、どんなネットワークもどんなアルゴリズムに対応づけられてしまう。ランダムなLMでも IOI で正答率100%になる。
  → 対応づけに使う写像のクラスを、型で制限する必要がある。

### 5.2 局在の落とし穴

- **[Is This the Subspace You Are Looking for?](https://arxiv.org/abs/2311.17030)**（Makelov et al., ICLR 2024）[PDF](../third_party/papers/representations/makelov2024_subspace_illusion.pdf)
  部分空間へのパッチが、眠っていた別経路を活性化して、意図した効果に見えることがある。MLP の1次元の活性パッチは、そのトークンで同じ出力を出すランク1編集に必ず書き換えられる。そのため、事実がそこに保存されていなくても編集は「成功」する。
  → Hase et al. の結果の説明。局在は候補を出すのに使い、成否は検証で判断する。

- **[Mechanistic Unlearning](https://arxiv.org/abs/2410.12949)**（Guo et al., ICML 2025）[PDF](../third_party/papers/representations/guo2025_mechanistic_localization.pdf)
  事実を引き出す仕組みに局在させた編集は、出力を追跡して局在させた編集より頑健で、副作用も少ない。
  → 局在が編集の役に立つかは、局在の方法による。

### 5.3 特徴の辞書

- **[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)**（Bricken et al., 2023, Web記事）、**[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)**（Templeton et al., 2024, Web記事）
  疎オートエンコーダで、活性値を解釈できる特徴の和に分解する。後者は Claude 3 Sonnet で最大3400万特徴。著者自身、目的関数は代理指標にすぎず、すべての特徴を見つけるには桁違いに足りないと述べる。

- **[Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)**（Ameisen et al., 2025, Web記事）、**[On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)**（Lindsey et al., 2025, Web記事）
  層をまたぐトランスコーダで置き換えモデルを作り、プロンプトごとの計算グラフを描く。同じ仕組みが使われている保証はなく、満足な洞察が得られたのは約4分の1のプロンプト。多言語の処理には、言語固有の回路と言語に依存しない回路が混在する。
  → 名前から方向への対応表に最も近いが、ラベルは事後的で不完全。広い能力は分散している。

- **[AxBench](https://arxiv.org/abs/2501.17148)**（Wu et al., ICML 2025）[PDF](../third_party/papers/representations/wu2025_axbench.pdf)
  概念の検出と、概念による出力の誘導を比べるベンチマーク。誘導ではプロンプトが最も良く、ファインチューニング（LoReFT、LoRA）が続き、疎オートエンコーダは単純な手法に及ばない。
  → 上位層（名前から K・V・S を作る）の評価の枠組みとして使える。

### 5.4 出力の誘導と、重みへの焼き込み

- **ActAdd**（[Turner et al., 2023](https://arxiv.org/abs/2308.10248)）[PDF](../third_party/papers/representations/turner2023_actadd.pdf)、**ITI**（[Li et al., NeurIPS 2023](https://arxiv.org/abs/2306.03341)）[PDF](../third_party/papers/representations/li2023_iti.pdf)、**CAA**（[Panickssery (Rimsky) et al., ACL 2024](https://arxiv.org/abs/2312.06681)）[PDF](../third_party/papers/representations/panickssery2024_caa.pdf)、**RepE**（[Zou et al., 2023](https://arxiv.org/abs/2310.01405)）[PDF](../third_party/papers/representations/zou2023_repe.pdf)
  活性値の差分ベクトルを足して出力を誘導する。ITI は、一定のベクトルを足す介入をバイアス項の書き換えとして重みに焼き込めると述べる。RepE の LoRRA は、表現の目標に向けて LoRA を学習する。

- **[Function Vectors](https://arxiv.org/abs/2310.15213)**（Todd et al., ICLR 2024）[PDF](../third_party/papers/representations/todd2024_function_vectors.pdf)、**[In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916)**（Hendel et al., EMNLP 2023 Findings）[PDF](../third_party/papers/representations/hendel2023_icl_task_vectors.pdf)
  「タスク＝ベクトル」は、狭いタスクについて、活性値のレベルで、特定の位置でなら成り立つ。

- **[Persona Vectors](https://arxiv.org/abs/2507.21509)**（Chen et al., 2025）[PDF](../third_party/papers/representations/chen2025_persona_vectors.pdf)
  性格特性の説明から、LLMが対照的なプロンプト・質問・採点基準を作り、平均の差として方向を自動で得る。ファインチューニング中に誘導をかけると、性格の変化を防げる。システムプロンプトで誘発できる特性に限られる。
  → 「名前 → 方向」を自動化した例。

- **[AlphaSteer](https://arxiv.org/abs/2506.07022)**（Sheng et al., ICLR 2026）[PDF](../third_party/papers/representations/sheng2026_alphasteer.pdf)
  活性値の誘導を、無害な活性値に対する零空間制約つきの正則化最小二乗として閉形式で解く。
  → 活性値のレベルで、`Soft` の写像と `Exact(U)` の保存とリッジを組み合わせた編集と同じ形の問題。

- **[Concept Ablation Fine-Tuning (CAFT)](https://arxiv.org/abs/2507.16795)**（Casademunt et al., ICML 2026）[PDF](../third_party/papers/representations/casademunt2026_caft.pdf)
  PCA や疎オートエンコーダで見つけた方向を、ファインチューニング中だけ取り除き、学習される更新を制約する。創発的な不整合が 7.0% から 0.51% に下がる。
  → 解釈できる部分空間を、学習則の制約（`Rule(preserve=...)`）として使う先例。

- **[Steering Language Models with Weight Arithmetic](https://arxiv.org/abs/2511.05408)**（Fierro & Roger, ICLR 2026）[PDF](../third_party/papers/representations/fierro2026_weight_arithmetic_steering.pdf)
  逆向きの小さなファインチューニング2つの差から、重み空間での振る舞いの方向を得る。活性値の誘導より分布外に汎化しやすい。

- **[Weight Updates as Activation Shifts](https://proceedings.mlr.press/v306/adila26a.html)**（Adila et al., ICML 2026）[PDF](../third_party/papers/representations/adila2026_weight_updates_activation_shifts.pdf)
  活性値の誘導と重みの更新は1次の範囲でしか等価でなく、表せる関数のクラスが違う。
  → 活性値の介入を重みに厳密に変換できるのは特殊な場合（一定ベクトル＝バイアス、方向の除去＝直交化）だけ。言語は、どの部分クラスを扱うのかを明示する。

### 5.5 内部への介入を宣言的に書く道具

- **[pyvene](https://arxiv.org/abs/2403.07809)**（Wu et al., NAACL 2024 Demo）[PDF](../third_party/papers/representations/wu2024_pyvene.pdf)
  介入を設定（層、部品、介入の種類、低ランクの次元）として宣言的に書き、保存できる。

- **[NNsight and NDIF](https://arxiv.org/abs/2407.14561)**（Fiotto-Kaufman et al., ICLR 2025）[PDF](../third_party/papers/representations/fiottokaufman2025_nnsight.pdf)
  介入を遅延評価の「介入グラフ」として記録し、JSONにしてリモートで実行する。

- **[ReFT](https://arxiv.org/abs/2404.03592)**（Wu et al., NeurIPS 2024）[PDF](../third_party/papers/representations/wu2024_reft.pdf)
  `Φ(h) = h + Rᵀ(Wh + b − Rh)`（R の行は正規直交）。部分空間の中だけで表現を書き換える、ファインチューニングの部品。

  → いずれも活性値レベルのDSLに近いものだが、意味の型・制約・ソルバ・保証は持たない。モデルの調査や統計量の収集では、実装の参考にする。

### 構想への含意

- 「名前 → 部分空間」には形式的な意味論（因果抽象化）があるが、写像のクラスを制限しないと意味がなくなる。
- 数学能力や日本語能力のような広い能力を、保証つきで部分空間に結びつけた例はない。上位層は「候補を出して検証する」探索として設計する。
- 計量（内積）の選び方は自明ではない。モデルと層ごとに確かめる。

---

## 6. LLM／探索による学習則・プログラム・重みの合成（synthesis）

### 6.1 学習済み最適化器

- **[VeLO](https://arxiv.org/abs/2211.09760)**（Metz et al., 2022）[PDF](../third_party/papers/synthesis/metz2022_velo.pdf)
  テンソルごとの LSTM が、パラメータごとの小さな MLP を生成し、更新量を出す。約4000 TPU月かけて進化戦略でメタ学習した。著者は、約5億パラメータを超えるモデル、20万ステップ以上の学習、強化学習などで失敗すると報告している。更新則を記号で選ぶのではなく、数値を出す。
  → このリポジトリの出発点。比較対象（`third_party/learned_optimization`）。

- **[Is Scaling Learned Optimizers Worth It?](https://arxiv.org/abs/2310.18191)**（Rezk et al., 2023）[PDF](../third_party/papers/synthesis/rezk2023_velo_evaluation.pdf)
  VeLO の総ステップ数のヒントは問題ごとに調整が要る重要なハイパーパラメータで、MLCommons のワークロードでは既存の最適化器より速くなかった。

### 6.2 プログラムの探索

- **[AutoML-Zero](https://arxiv.org/abs/2003.03384)**（Real et al., ICML 2020）[PDF](../third_party/papers/synthesis/real2020_automl_zero.pdf)
  65個の基本的な数学演算から、学習アルゴリズム全体を進化させる。受け入れられるプログラムは 10⁷〜10¹² 個に1個程度。等価なプログラムの指紋と足切りで探索を速める。
  → 基本演算のDSLの上での探索の原型。

- **[Symbolic Discovery of Optimization Algorithms (Lion)](https://arxiv.org/abs/2302.06675)**（Chen et al., NeurIPS 2023）[PDF](../third_party/papers/synthesis/chen2023_lion.pdf)
  NumPy 風の45関数で最適化器のプログラムを進化させた。抽象実行で型と形を推論して不正なプログラムを除き、意味のハッシュで重複を省く。探索に使った小さなタスクと目標のタスクの差（1万倍以上）が過学習を生み、段階的に大きなタスクで選抜した。条件分岐やループは役に立たなかった。
  → 型検査そのものは新規性にならない。代理タスクから目標タスクへの汎化が共通の難所。

- **[FunSearch](https://www.nature.com/articles/s41586-023-06924-6)**（Romera-Paredes et al., Nature 2023）[PDF](../third_party/papers/synthesis/romeraparedes2023_funsearch.pdf)、**[AlphaEvolve](https://arxiv.org/abs/2506.13131)**（Novikov et al., 2025）[PDF](../third_party/papers/synthesis/novikov2025_alphaevolve.pdf)
  LLM がプログラムを書き換え、評価器が採点する進化的探索。FunSearch は約100万サンプルを使い、140回の実行のうち4回だけが目標に到達した。AlphaEvolve は勾配法による行列分解の手順全体を進化させ、Gemini の学習時間を約1%短縮するカーネルも見つけた。どちらも自動で採点できる評価器が前提。
  → 探索エンジンの比較対象（`third_party/ShinkaEvolve`、`third_party/openevolve`）。

- **[DiscoPOP](https://arxiv.org/abs/2406.08414)**（Lu et al., NeurIPS 2024）[PDF](../third_party/papers/synthesis/lu2024_discopop.pdf)
  GPT-4 が選好最適化の損失関数のコードを書き、短いファインチューニングで評価する。見つけた損失は、探索時と違う β では収束しにくい。

- **[Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2309.03409)**（Yang et al., ICLR 2024）[PDF](../third_party/papers/synthesis/yang2024_opro.pdf)
  過去の解と点数を見せて、LLM に次の解を出させる。線形回帰では係数を直接出す。勾配法に勝つことは目的としておらず、問題が大きいと劣化する。

- **[Automated Design of Agentic Systems (ADAS)](https://arxiv.org/abs/2408.08435)**（Hu et al., ICLR 2025）[PDF](../third_party/papers/synthesis/hu2025_adas.pdf)
  メタエージェントがエージェントを Python で書き、アーカイブにためていく。

- **[How Should We Meta-Learn Reinforcement Learning Algorithms?](https://arxiv.org/abs/2507.17668)**（Goldie et al., RLC 2025）[PDF](../third_party/papers/synthesis/goldie2025_meta_learn_rl_algorithms.pdf)
  LLM に学習則のコードを書かせる方法を、進化戦略や蒸留と比べる。入力が少なく良い初期値があるときは効くが、入力が多いと、すべての入力を使う良いアルゴリズムを LLM が提案するのは難しい。

- **[OPTScientist](https://arxiv.org/abs/2607.20486)**（Li et al., 2026 プレプリント）[PDF](../third_party/papers/synthesis/li2026_optscientist.pdf)
  複数のLLMエージェントが、型つきのDSLで最適化器のプログラムを書き、PyTorch にコンパイルする。DSL をマクロで拡張することもできる。11億パラメータのモデルの事前学習で、Muon に近い性能の最適化器を見つけた。
  → 学習則の側で最も近い先行研究。制約の意味論と、言語の比較実験はない。

### 6.3 LLMが仕様やデータを書く／重みを生成する

- **[Self-Adapting Language Models (SEAL)](https://arxiv.org/abs/2506.10943)**（Zweiger et al., NeurIPS 2025）[PDF](../third_party/papers/synthesis/zweiger2025_seal.pdf)
  モデルが自分用の「自己編集」（知識タスクでは学習用の文章、ARC ではデータ拡張の指定と学習率・エポック数など）を書き、LoRA による SFT で適用する。「適用後に良くなったか」を報酬にして、自己編集の書き方を学習する。逐次的な自己編集で忘却が起きる。
  → 「LLM が仕様を書き、数値計算がそれを実行する」構図に最も近い。仕様はデータとハイパーパラメータで、制約も保証もない。

- **[Text-to-LoRA](https://arxiv.org/abs/2506.06105)**（Charakorn et al., ICML 2025）[PDF](../third_party/papers/synthesis/charakorn2025_text_to_lora.pdf)、**[Drag-and-Drop LLMs](https://arxiv.org/abs/2506.16406)**（Liang et al., NeurIPS 2025）[PDF](../third_party/papers/synthesis/liang2025_drag_and_drop.pdf)
  タスクの説明やプロンプトから、ハイパーネットワークが LoRA の重みを直接作る。
  → 安価で強い比較対象。

- **[NeuroGen](https://arxiv.org/abs/2505.12470)**（Wang et al., 2025）[PDF](../third_party/papers/synthesis/wang2025_neurogen.pdf)
  LLM に小さなネットワークの重みを直接出力させる。CIFAR-10 で、普通に学習したCNNより大きく劣る。
  → LLMに重みを直接書かせない、という出発点の裏づけ。

### 6.4 エージェントとベンチマーク

- **[A Multimodal Automated Interpretability Agent (MAIA)](https://arxiv.org/abs/2404.14394)**（Rott Shaham et al., ICML 2024）[PDF](../third_party/papers/synthesis/shaham2024_maia.pdf)
  LLM が、モデルの部品を調べる道具群のAPIに対して Python を書き、実験する。重みは直接書き換えない。確証バイアスと道具の信頼性が課題。
  → 「LLMがモデル内部の型つきAPIに対してプログラムを書く」先例。

- **[PostTrainBench](https://arxiv.org/abs/2603.08640)**（Rank et al., 2026 プレプリント）[PDF](../third_party/papers/synthesis/rank2026_posttrainbench.pdf)
  エージェントに H100 10時間でモデルの事後学習をさせる。最良でも公式の instruct モデルに大きく劣り、テストデータで学習するなどの報酬ハッキングが起きた。
  → エージェントが触れられない保留テストセットが必須。

- **[AI4AI-Bench](https://arxiv.org/abs/2608.20318)**（Chi et al., 2026 プレプリント）[PDF](../third_party/papers/synthesis/chi2026_ai4ai_bench.pdf)
  エージェントが学習アルゴリズムのコード（モデルスープ、枝刈り、アンラーニングなど）を書き換えるベンチマーク。正規化した平均点は低く、多くのエージェントは学習の仕組みそのものを変えない。

### 6.5 部品の自動獲得

- **[DreamCoder](https://arxiv.org/abs/2006.08381)**（Ellis et al., PLDI 2021）[PDF](../third_party/papers/synthesis/ellis2021_dreamcoder.pdf)
  問題を解きながら、解に共通する部分を新しい抽象（部品）として取り出す。

- **[Stitch: Top-Down Synthesis for Library Learning](https://arxiv.org/abs/2211.16605)**（Bowers et al., POPL 2023）[PDF](../third_party/papers/synthesis/bowers2023_stitch.pdf)
  DreamCoder より3〜4桁速く、部品を抽出する。

- **[LILO](https://arxiv.org/abs/2310.19791)**（Grand et al., ICLR 2024）[PDF](../third_party/papers/synthesis/grand2024_lilo.pdf)
  LLM と Stitch を組み合わせ、抽出した部品に名前と説明を付ける。名前のない部品は LLM の成績を下げ、名前と説明を付けると改善した。
  → 部品ライブラリは、圧縮率ではなく LLM の成績で評価し、名前と説明を必ず付ける。

### 構想への含意

- 「LLM＋評価器でコードを進化させる」仕組みと、その型検査はすでにある。制約つき最適化を意味論に持つ言語はない。
- 比較実験では、同じバックエンドを使う素の Python エージェント、ランダム探索や進化的探索、学習済み編集器（MEND、MALMEN）、ハイパーネットワーク（Text-to-LoRA）、SEAL 型の仕様を対照に置く。
- 費用は、無効な候補・評価の回数・シード数まで含めて数える。代理タスクから目標タスクへの選抜を段階的に行う。

---

## 7. 制約つき学習と更新の幾何（optimization）

### 7.1 保存部分空間を学習に組み込む

- **[Orthogonal Gradient Descent](https://arxiv.org/abs/1910.07104)**（Farajtabar et al., AISTATS 2020）[PDF](../third_party/papers/optimization/farajtabar2020_ogd.pdf)
  過去タスクの出力の勾配を保存し、新しい勾配をそれらと直交化する。保証は1次近似のみで、学習率が大きいと崩れる。

- **[Gradient Projection Memory (GPM)](https://arxiv.org/abs/2103.09762)**（Saha et al., ICLR 2021）[PDF](../third_party/papers/optimization/saha2021_gpm.pdf)
  層ごとに活性値のSVDをとり、エネルギーの閾値で部分空間を決め、勾配をその直交補空間へ射影する。
  → `ΔW U = 0` をSGDにコンパイルしたもの。閾値つきの意味論がはっきりしている。

- **[Adam-NSCL](https://arxiv.org/abs/2103.07113)**（Wang et al., CVPR 2021）[PDF](../third_party/papers/optimization/wang2021_adam_nscl.pdf)
  特徴の（中心化しない）共分散の零空間を、最小固有値の数倍を閾値として近似する。Adam の更新を計算した**後で**射影する。厳密な零空間を仮定すれば、層ごとの帰納法で過去の出力が保たれる。
  → 「射影してから Adam」では制約が壊れる、という交換則の実例。

- **[O-LoRA](https://arxiv.org/abs/2310.14152)**（Wang et al., EMNLP 2023 Findings）[PDF](../third_party/papers/optimization/wang2023_olora.pdf)、**[InfLoRA](https://arxiv.org/abs/2404.00228)**（Liang & Li, CVPR 2024）[PDF](../third_party/papers/optimization/liang2024_inflora.pdf)
  O-LoRA はタスク間の LoRA 部分空間の直交性をペナルティで促す（機能的な保証はない）。InfLoRA は、新しいタスクの入力と過去の勾配空間の直交補空間から B を設計して固定し、A だけを学習する。「保存＋ランク上限」を再パラメータ化で守るので、最適化器に依存しない。

- **[Put the Space of LoRA Initialization to the Extreme (LoRA-Null)](https://arxiv.org/abs/2503.02659)**（Tang et al., AAAI 2026）[PDF](../third_party/papers/optimization/tang2026_lora_null.pdf)
  事前学習の知識に対応する活性値の零空間から LoRA を初期化する。

- **[Null-Space Constrained Policy Optimization (NSPO)](https://arxiv.org/abs/2512.11391)**（Niu et al., ICLR 2026）[PDF](../third_party/papers/optimization/niu2026_nspo.pdf)
  AlphaEdit 型の射影を、汎用タスクの活性値から作り、GRPO による安全性の学習の勾配にかける。射影した勾配がまだ降下方向であることを示す（素の勾配降下について）。

### 7.2 低ランク構造

- **[LoRA](https://arxiv.org/abs/2106.09685)**（Hu et al., ICLR 2022）[PDF](../third_party/papers/optimization/hu2022_lora.pdf)
  `ΔW = BA` で、学習全体を通してランクが r 以下に保たれる。

- **[Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning](https://arxiv.org/abs/2012.13255)**（Aghajanyan et al., ACL 2021）[PDF](../third_party/papers/optimization/aghajanyan2021_intrinsic_dimension.pdf)
  全パラメータのランダムな200次元部分空間で、フルのファインチューニングの90%の性能に届くタスクがある。行列ごとのランクではなく、全体の部分空間の話。

- **[GaLore](https://arxiv.org/abs/2403.03507)**（Zhao et al., ICML 2024）[PDF](../third_party/papers/optimization/zhao2024_galore.pdf)
  勾配の上位r成分の部分空間に射影し、一定ステップごとに部分空間を取り直す。累積した更新のランクは制限されない。「保存」とは逆に、支配的な部分空間を残す。
  → `Rule(rank=...)` が抑えるのはステップごとのランクで、累積したランクではない。

### 7.3 ノルムと計量の理論

- **[K-FAC](https://arxiv.org/abs/1503.05671)**（Martens & Grosse, ICML 2015）[PDF](../third_party/papers/optimization/martens2015_kfac.pdf)
  Fisher 行列を、入力活性値の共分散と出力勾配の共分散のクロネッカー積で近似する。減衰やモメンタムがなければ、活性値の可逆なアフィン変換に対して更新の軌跡が不変。
  → `Rule(metric="covariance")` と AgentOptimizer の `ctx.covariance` は、K-FAC の入力側の因子だけを使った計量（ROME・MEMIT の C⁻¹ も同じ）。出力側の因子も入れた `tr(G_out ΔW C_in ΔWᵀ)` は未実装。

- **[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)**（Bernstein & Newhouse, OPT 2024 Workshop）[PDF](../third_party/papers/optimization/bernstein2024_old_optimizer_new_norm.pdf)
  モメンタムなどを外すと、Adam は ℓ∞ 型のノルム、Shampoo はスペクトルノルム、SGD はフロベニウスノルムのもとでの最急降下になる。
  → 「計量 → 最適化器」の辞書。

- **[Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813)**（Large et al., NeurIPS 2024）[PDF](../third_party/papers/optimization/large2024_modular_norm.pdf)
  モジュールを（順伝播、質量、感度、ノルム）の組とし、合成の規則でネットワーク全体のノルムを作る。「よく正規化されている」という条件は型付け判定のように働く。

- **[Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)**（Bernstein & Newhouse, 2024）[PDF](../third_party/papers/optimization/bernstein2024_modular_duality.pdf)
  勾配は双対ベクトルなので、更新に使うには「双対化」が要る。線形層の双対化は `√(d_out/d_in)·UVᵀ`（Muon に対応）。論文自身が「深層学習の型システム」を提案している。
  → 構想の「計量の明示」に理論の土台を与える。ただしデータに依存しない（C を持たない）。

- **[Muon](https://kellerjordan.github.io/posts/muon/)**（Jordan, 2024, ブログ）、**[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)**（Liu et al., 2025）[PDF](../third_party/papers/optimization/liu2025_muon_scalable.pdf)
  モメンタムの後に Newton–Schulz 反復で直交化する。後者は重み減衰と更新の大きさの調整を加え、AdamW の約2倍の計算効率を報告。事前学習と SFT で最適化器が違うと、効果がはっきりしない。

- **[Training Deep Learning Models with Norm-Constrained LMOs (Scion)](https://arxiv.org/abs/2502.07529)**（Pethick et al., ICML 2025）[PDF](../third_party/papers/optimization/pethick2025_scion.pdf)
  層ごとの作用素ノルム球の上の線形最小化で、正規化SGD・符号SGD・Muon をまとめる。重み減衰はノルムの上限に対応する。

- **[Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/)**（Bernstein, 2025, ブログ）
  重みを多様体の上に制約した Muon（manifold Muon）。各ステップで制約つきの凸問題を双対上昇法で解く。
  → 「コンパイラが制約つき最適化問題に変換する」に最も近い、最適化器の側の例。

- **[Training Transformers with Enforced Lipschitz Constants](https://arxiv.org/abs/2507.13338)**（Newhouse et al., 2025）[PDF](../third_party/papers/optimization/newhouse2025_lipschitz_transformers.pdf)
  重みのスペクトルノルムの制約を学習中ずっと守る。Muon との組み合わせが最も良い。

- **[μLO](https://arxiv.org/abs/2406.00153)**（Thérien et al., ICLR 2026）[PDF](../third_party/papers/optimization/therien2026_mulo.pdf)
  VeLO を含む学習済み最適化器は、メタ学習時より幅の広いネットワークで性能が落ちる。μP を導入すると大きく改善する。
  → 学習済みの更新則にも、型つきのスケーリング規則が要る。

### 構想への含意

- 同じ `preserve(S)` でも、最適化器によって正しい変換が違う（SGD と Muon ではそのまま射影すれば保たれ、Adam や符号化では後で射影する必要があり、重み減衰も射影する）。`Rule` は、方向の変換の後で射影し、重み減衰にも同じ射影をかける。
- 計量の理論は2系統ある。データに依存せず合成できる modular norm と、データに依存し座標変換に不変な K-FAC／C計量。両者を統一した研究は見つからなかった。
- 最適化器の側で、ユーザーの制約を受け取る学習済み・LLM製の更新則は見つからなかった。

---

## 8. 比較対象（ベースライン）

| 種類 | 比較対象 | 参照実装（`third_party/`） |
|---|---|---|
| 固定ルールの編集 | ROME、MEMIT、EMMET、AlphaEdit | `rome`、`memit`、`unified-model-editing`、`AlphaEdit`、`EasyEdit` |
| 学習済みの編集器 | MEND、MALMEN | `EasyEdit` |
| ハイパーネットワーク | Text-to-LoRA | `text-to-lora` |
| LLMが仕様を書く | SEAL 型（データ＋ハイパーパラメータ） | `SEAL` |
| 同じバックエンドを使うエージェント | 素の Python、JSON 仕様、ルールベース | （このリポジトリで実装する。未着手） |
| 進化的探索 | ShinkaEvolve、OpenEvolve | `ShinkaEvolve`、`openevolve` |
| 最適化器 | Adam・SGD・Muon（分類と CartPole で比較済み、[experiments.md](experiments.md)）、AdamW、VeLO | `Muon`、`modula`、`learned_optimization` |
| 評価データ | CounterFact・zsRE、RippleEdits、MQuAKE、合成の評価 | `EasyEdit`、`RippleEdits`、`MQuAKE`、`composable-interventions` |
| 正解のある環境 | Tracr、InterpBench | `tracr`、`InterpBench` |
