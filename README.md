# Deep Learning Studio

自分で入力した文章だけを使い、**BPEトークナイザーとGPT型言語モデルをゼロから学習**させるローカル学習アプリです。学習済みLLMや外部APIを呼び出して結果だけを表示するものではありません。PyTorchで実装した因果的自己注意（Causal Self-Attention）とTransformerを、CPU・CUDA・Apple Silicon（MPS）で実際に訓練します。

## 見えるもの

- 入力データからBPEの語彙が作られていく過程
- `文字列 → Token ID → Embedding → Self-Attention → FFN → 語彙確率` のモデル構造
- 学習中のLoss、検証Loss、Perplexity、学習率、勾配ノルム
- 同じプロンプトに対する学習途中の予測変化
- 各Attention Headが、どのトークンを参照したかを示すヒートマップ
- 文章生成時の上位候補トークンと確率

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

PyTorchは実行環境に応じてCUDA/MPS/CPUを自動選択します。NVIDIA GPU向けPyTorchの導入方法は環境ごとに異なるため、必要なら [PyTorch公式インストール案内](https://pytorch.org/get-started/locally/) のコマンドで先にPyTorchを入れてください。

## 最初の実験

1. 「データと設定」へ、同じ文体・話題を持つ文章を貼り付けます。
2. 「トークナイザーを構築」を押し、語彙が作られる様子を確認します。
3. 小型モデル設定を選び、「学習を開始」を押します。
4. グラフとAttentionを見ながらLossが下がる様子を観察します。
5. 「文章生成」で学習データに含まれる書き出しを入力し、続きを生成します。

数行でも動作しますが、言語らしい出力には最低でも数千文字、できれば数万文字以上が必要です。入力したデータは外部送信されません。

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
