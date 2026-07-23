# VLM: From Attention to Qwen

**Attention から Qwen-VL まで、Vision-Language Model（VLM）の進化を「実装しながら」追って学ぶ教材です。**

教材は 2 つの形で用意しています。どちらも中身は同じ実装です:

1. **`notebooks/`** — [marimo](https://marimo.io)（リアクティブな Python ノートブック）。各章が自己完結し、
   図とともに対話的に読める。**学ぶための入口**。
2. **`vlm/`** — 同じロジックを整理した**再利用可能な NumPy パッケージ**（`import vlm`）。テスト付き。
   **コードとして使う・読む・改造するための本体**。

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

`vlm/` パッケージも各章に対応しています:

| モジュール | 対応章 | 主な API |
|-----------|--------|---------|
| `vlm.attention` | 01 | `scaled_dot_product_attention`, `multi_head_attention`, `causal_mask` |
| `vlm.transformer` | 02 | `layer_norm`, `feed_forward`, `sinusoidal_positional_encoding`, `TinyGPT` |
| `vlm.vit` | 03 | `patchify`, `VisionTransformer` |
| `vlm.clip` | 04 | `l2_normalize`, `cosine_sim_matrix`, `clip_loss`, `zero_shot_classify` |
| `vlm.multimodal` | 05 | `simple_vision_encoder`, `build_projector`, `assemble_sequence`, `run_llm` |
| `vlm.qwen` | 06 | `apply_rope`, `assign_mrope_positions`, `apply_mrope`, `patch_merger`, `n_vision_tokens` |

---

## プロジェクト構成

```
vlm-from-attention-to-qwen/
├── notebooks/          学習用 marimo ノートブック（01〜06、各章自己完結）
├── vlm/                再利用可能な NumPy パッケージ（import vlm）
│   ├── attention.py    transformer.py  vit.py
│   └── clip.py         multimodal.py   qwen.py
├── tests/              pytest（各モジュールの shape と性質を検証）
└── pyproject.toml
```

---

## 使い方

### セットアップ

```bash
# ノートブックも動かす場合（marimo/matplotlib/pillow 込み）
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks]"

# ライブラリだけ使う場合（NumPy のみ）
pip install -e .

# テストも走らせる場合
pip install -e ".[dev]"
```

### ノートブックを開く（編集モード・リアクティブ）

```bash
marimo edit notebooks/01_attention.py
```

セルを編集すると依存するセルが自動で再計算されます。まずはコードを読み、値を変えて挙動を観察してください。
読み物として実行するだけなら `marimo run notebooks/01_attention.py`。

### ライブラリとして使う

```python
import numpy as np
from vlm.attention import scaled_dot_product_attention, causal_mask
from vlm.qwen import apply_rope, assign_mrope_positions, apply_mrope

rng = np.random.default_rng(0)
X = rng.standard_normal((5, 8))
out, weights = scaled_dot_product_attention(X, X, X, mask=causal_mask(5))

# Qwen-VL の M-RoPE を試す
pos, seg = assign_mrope_positions(n_text_before=2, grid_hw=(4, 4), n_text_after=3)
rotated = apply_mrope(rng.standard_normal((len(seg), 36)), pos)
```

### テスト / 動作確認

```bash
pytest                      # vlm/ パッケージのテスト（29 件）

# 全ノートブックが実行できるかの確認
for f in notebooks/*.py; do python -m marimo export html "$f" -o /dev/null \
  && echo "OK $f" || echo "FAILED $f"; done
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
