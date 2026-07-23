# VLM: From Attention to Qwen

**Attention から Qwen-VL まで、Vision-Language Model（VLM）の進化を「実装しながら」追って学ぶ教材です。**

各ステップを [marimo](https://marimo.io)（リアクティブな Python ノートブック）で用意しました。
すべて **NumPy でゼロから実装** しているので、フレームワークの中身に隠れず、行列演算のレベルで仕組みが見えます。
重い GPU も学習済みモデルのダウンロードも不要（最終章のオプションを除く）。CPU で数秒で動きます。

> なぜ NumPy か？ PyTorch の `nn.MultiheadAttention` を呼ぶだけでは「動くけど分からない」ままです。
> ここでは QKV の行列積・softmax・マスクを一つずつ手で書き、後半で「これが PyTorch / transformers のどこに当たるか」を対応づけます。

---

## 学習パス

```
                    ┌─────────────────────────────────────────────────────┐
                    │  テキストの世界                    画像の世界          │
                    └─────────────────────────────────────────────────────┘

  01 Attention ──▶ 02 Transformer ──┬──────────────▶ 03 Vision Transformer
   注意機構が          言語モデルの     │  「画像も             画像をパッチ＝
   全ての基礎          基本ブロック    │   トークン列」        トークン列にする
                                      │
                                      ▼
                          04 CLIP  ◀──┘
                    画像とテキストを
                    同じ空間に並べる（対照学習）
                            │
                            ▼
              05 VLM (LLaVA スタイル)
              画像エンコーダ → projector → LLM
              「画像を見て言葉で答える」
                            │
                            ▼
              06 Qwen-VL の工夫
              動的解像度 / M-RoPE / merger
              → 本物の Qwen3-VL への橋渡し
```

| # | ノートブック | 学ぶこと | キーワード |
|---|-------------|---------|-----------|
| 01 | `notebooks/01_attention.py` | 注意機構をゼロから | scaled dot-product, causal mask, multi-head |
| 02 | `notebooks/02_transformer.py` | Transformer ブロック | LayerNorm, FFN, 残差, 位置エンコーディング |
| 03 | `notebooks/03_vision_transformer.py` | 画像をトークンに | パッチ埋め込み, CLS トークン, ViT |
| 04 | `notebooks/04_clip.py` | 画像とテキストを繋ぐ | 対照学習, コサイン類似度, ゼロショット |
| 05 | `notebooks/05_vlm_llava.py` | 見て答えるモデル | vision encoder, projector, トークン結合 |
| 06 | `notebooks/06_qwen_vl.py` | Qwen-VL の中核技術 | 動的解像度, M-RoPE, vision merger |

各ノートブックは前の章の結論だけ前提にして、独立して動きます。順に進めるのがおすすめです。

---

## 使い方

### セットアップ

```bash
# uv を使う場合（推奨）
uv venv && source .venv/bin/activate
uv pip install -e .

# もしくは pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### ノートブックを開く（編集モード・リアクティブ）

```bash
marimo edit notebooks/01_attention.py
```

セルを編集すると依存するセルが自動で再計算されます。まずはコードを読み、値を変えて挙動を観察してください。

### アプリとして実行する（読み物モード）

```bash
marimo run notebooks/01_attention.py
```

### 全ノートブックの動作確認

```bash
for f in notebooks/*.py; do
  echo "== $f =="
  python -m marimo export html "$f" -o /dev/null || echo "FAILED: $f"
done
```

---

## この教材の考え方

1. **小さく作る** — 語彙数十・次元数個の「おもちゃ」サイズで動かします。数式の shape が頭に入るのが目的で、性能ではありません。
2. **可視化する** — attention の重み、パッチ分割、類似度行列などを図で確認します。
3. **本物に繋げる** — 各章末に「PyTorch / Hugging Face transformers ではどう書くか」「実際の Qwen3-VL のどのコンポーネントか」を対応づけます。
4. **一本の物語にする** — Attention → Transformer → ViT → CLIP → VLM → Qwen-VL は独立した話ではなく、同じ注意機構が形を変えて積み上がっていく一本道です。

---

## 参考文献（原典）

- **Attention Is All You Need** — Vaswani et al., 2017（Transformer）
- **An Image is Worth 16x16 Words** — Dosovitskiy et al., 2020（ViT）
- **Learning Transferable Visual Models from Natural Language Supervision** — Radford et al., 2021（CLIP）
- **Visual Instruction Tuning** — Liu et al., 2023（LLaVA）
- **Qwen2-VL / Qwen2.5-VL Technical Report** — Qwen Team, 2024–2025（動的解像度・M-RoPE）

---

## この教材を独立したリポジトリにする

このディレクトリは自己完結しています。単体のリポジトリに切り出すには：

```bash
# 1. GitHub で空リポジトリ vlm-from-attention-to-qwen を作成
# 2. このディレクトリだけを取り出して push
cd vlm-from-attention-to-qwen
git init
git add .
git commit -m "VLM tutorial: from attention to qwen"
git branch -M main
git remote add origin git@github.com:<you>/vlm-from-attention-to-qwen.git
git push -u origin main
```
