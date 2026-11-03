# Deep Learning Studio

自分で入力した文章だけを使い、**BPEトークナイザーとGPT型言語モデルをゼロから学習**させるローカル学習アプリです。学習済みLLMや外部APIを呼び出して結果だけを表示するものではありません。PyTorchで実装した因果的自己注意（Causal Self-Attention）とTransformerを、CPU・CUDA・Apple Silicon（MPS）で実際に訓練します。

## 見えるもの

- 入力データからBPEの語彙が作られていく過程
- `文字列 → Token ID → Embedding → Self-Attention → FFN → 語彙確率` のモデル構造
- 学習中のLoss、検証Loss、Perplexity、学習率、勾配ノルム
- 同じプロンプトに対する学習途中の予測変化
- 各Attention Headが、どのトークンを参照したかを示すヒートマップ
- 文章生成時の上位候補トークンと確率
- EOS・句点・改行による自然な長さの自動停止（最大Token数の安全上限あり）

## セットアップ

Python 3.10以上を推奨します。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
python -m deep_learning_studio
```

### Windows + NVIDIA GPU（RTXシリーズ）

通常の `pip install torch` でCUDAを含まないPyTorchが入ると、NVIDIA GPUを搭載していてもCUDAは利用できません。このプロジェクトには、CUDA版PyTorchを専用仮想環境へ導入して検証するPowerShellスクリプトがあります。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_windows_cuda.ps1
.venv\Scripts\python -m deep_learning_studio
```

標準では互換性を重視してCUDA 12.6版を使用します。新しいドライバー環境では `-CudaVariant cu130` または `-CudaVariant cu132` も選択できます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_windows_cuda.ps1 -CudaVariant cu130
```

アプリの「GPU・アクセラレータ診断を表示」では、PyTorchの種類、CUDA Runtime、GPU名、NVIDIA Driver、利用できない理由を確認できます。PyTorchは実行環境に応じて次の順で自動選択します。

1. CUDA（NVIDIA）またはROCm（AMD）
2. XPU（Intel GPU）
3. MPS（Apple Silicon）
4. DirectML（Windows上の各社GPU、`torch-directml`導入時）
5. CPU

NVIDIA以外では、各バックエンドに対応したPyTorchを先に導入してください。DirectMLは `pip install torch-directml` で追加でき、アプリが自動検出します。最新の正式な導入方法は [PyTorch公式インストール案内](https://pytorch.org/get-started/locally/) を参照してください。

コマンドラインだけで診断することもできます。

```bash
python scripts/diagnose_accelerator.py
```

## 最初の実験

1. 「データと設定」へ、同じ文体・話題を持つ文章を貼り付けます。
2. 「トークナイザーを構築」を押し、語彙が作られる様子を確認します。
3. 小型モデル設定を選び、「学習を開始」を押します。
4. グラフとAttentionを見ながらLossが下がる様子を観察します。
5. 「文章生成」で学習データに含まれる書き出しを入力し、続きを生成します。

数行でも動作しますが、言語らしい出力には最低でも数千文字、できれば数万文字以上が必要です。入力したデータは外部送信されません。

## 継続学習と「日本語の習得」の観測

既存モデルの重みを維持したまま新しい日本語文章を学習するための「**現在のモデルに追加学習**」ボタンを追加しました。

1. 最初の文章でBPEを構築して学習し、チェックポイント（`.pt`）を保存します。
2. 「日本語の習得度テスト」に**学習に混ぜない**別の文章を入れます。学習中に日本語テストのPPLと経過秒数を表示します。
3. 新しい文章を入力し「現在のモデルに追加学習」を押します。前の重みとBPE語彙IDはそのまま維持し、未登録のUnicode文字だけを追加します。
4. 学習履歴はメニューからCSV出力できます。次回起動後もチェックポイントを開いて追加学習できます。

追加学習ではOptimizerを作り直し、Stepカウンターも今回のセッション用にリセットします。文字が新たに増えた場合、Embeddingと出力Headを拡張します。テスト文章の未登録文字がある場合はPPLの代わりに文字網羅率を表示します。

**PPLが低下することと日本語を理解したことは同義ではありません。** 未知の文章の生成品質や文法理解もあわせて確認してください。詳しい実験手順、比較上の注意点は [日本語の継続学習実験ガイド](docs/continued_learning.md) にまとめています。

## モデル規模の目安

| 設定 | 層 | 埋め込み | Head | 用途 |
|---|---:|---:|---:|---|
| 学習用ミニ | 2 | 64 | 4 | CPUで仕組みを観察 |
| 標準 | 4 | 128 | 4 | 小規模コーパスの実験 |
| 本格 | 6 | 192 | 6 | GPU推奨 |

これは教育用の「小型GPT」です。構造と計算はLLMと同系統ですが、ChatGPT級の能力には巨大なデータセット、計算資源、アラインメント工程が必要です。

## テスト

```bash
python -m unittest discover -s tests -v
```

詳細設計は [docs/architecture.md](docs/architecture.md) を参照してください。
