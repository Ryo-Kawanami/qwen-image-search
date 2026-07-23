import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 05. VLM (LLaVA スタイル) — 画像を LLM に「話させる」

        CLIP（04）は画像とテキストの**近さ**を測れました。でも文章は書けません。
        「この画像には何が写っていますか？」に**言葉で答える**には、
        視覚情報を **大規模言語モデル（LLM）** に流し込む必要があります。

        **LLaVA (Liu et al. 2023)** のレシピは驚くほど素直です:

        ```
          画像 ─[ 視覚エンコーダ (CLIP ViT, 凍結) ]─▶ パッチ特徴 (N × d_vision)
                                                          │
                                                   [ Projector (MLP) ]   ← ここだけ新規学習
                                                          │
                                                          ▼
                                            LLM の埋め込み空間の「画像トークン」 (N × d_llm)
                                                          │
          テキスト ─[ 語彙埋め込み ]─▶ テキストトークン ──┤
                                                          ▼
                            [ 画像トークン ＋ テキストトークン を1列に並べる ]
                                                          │
                                                   [ LLM (Transformer, 02) ]
                                                          │
                                                          ▼
                                                   次の単語を生成
        ```

        **鍵は projector**。「視覚エンコーダが出すベクトル」を「LLM が単語として受け取れるベクトル」へ翻訳する
        小さな橋です。LLaVA では視覚エンコーダと LLM は基本そのまま（凍結）で、**この橋を学習** するだけで
        画像が言葉になります。本章でその組み立てを NumPy で実装します。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(4)
    return np, plt, rng


@app.cell
def _(np):
    # --- 01〜03 の部品（LLM 本体に使う） ---
    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def sdpa(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)
        w = softmax(scores, axis=-1)
        return w @ V, w

    def mha(X, Wq, Wk, Wv, Wo, n_heads, mask=None, return_w=False):
        n, d = X.shape
        dh = d // n_heads
        split = lambda M: M.reshape(n, n_heads, dh).transpose(1, 0, 2)
        Qh, Kh, Vh = split(X @ Wq), split(X @ Wk), split(X @ Wv)
        outs, ws = [], []
        for h in range(n_heads):
            o, w = sdpa(Qh[h], Kh[h], Vh[h], mask)
            outs.append(o); ws.append(w)
        out = np.concatenate(outs, axis=-1) @ Wo
        return (out, np.stack(ws)) if return_w else out

    def layer_norm(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        return g * (x - mu) / np.sqrt(var + eps) + b

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def causal_mask(n):
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    return causal_mask, gelu, layer_norm, mha, softmax


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. 視覚エンコーダ（03 ViT の出力を模擬）

        03 の ViT は各パッチに対して `d_vision` 次元のベクトルを出しました。
        ここでは 03 と同じく画像をパッチ化し、ランダムな「視覚エンコーダ」でパッチ特徴を作ります
        （本物では CLIP 学習済み ViT）。ポイントは **CLS ではなく全パッチ特徴を使う**こと ——
        LLM には画像の細部まで渡したいからです。
        """
    )
    return


@app.cell
def _(np, rng):
    def make_image(size=32):
        H = W = size
        img = np.zeros((H, W, 3)); img[:, :, 2] = 0.3
        yy, xx = np.mgrid[0:H, 0:W]
        img[(xx - W * 0.4) ** 2 + (yy - H * 0.4) ** 2 < (size * 0.22) ** 2] = [0.9, 0.1, 0.1]
        img[(xx > W * 0.6) & (yy > H * 0.6)] = [0.95, 0.9, 0.1]
        return img

    def vision_encoder(img, P, d_vision, rng):
        """画像 → パッチ特徴 (n_patch, d_vision)。ランダム重み（本物は CLIP ViT）。"""
        H, W, C = img.shape
        gh, gw = H // P, W // P
        patches = [img[i*P:(i+1)*P, j*P:(j+1)*P, :].reshape(-1)
                   for i in range(gh) for j in range(gw)]
        patches = np.stack(patches)                      # (n_patch, P*P*C)
        Wenc = rng.standard_normal((patches.shape[1], d_vision)) / np.sqrt(patches.shape[1])
        return np.tanh(patches @ Wenc)                   # (n_patch, d_vision)

    image = make_image(32)
    patch_feats = vision_encoder(image, P=8, d_vision=24, rng=rng)  # 4x4=16 patches
    ("patch features:", patch_feats.shape)  # (16, 24)
    return image, patch_feats


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Projector — 視覚特徴を LLM の言語空間へ翻訳

        視覚特徴は `d_vision=24` 次元。LLM の埋め込みは `d_llm=32` 次元。次元も**意味**も違います。
        Projector（LLaVA では 2 層 MLP）がこのギャップを埋めます:

        $$\text{image\_token} = \mathrm{GELU}(\text{patch\_feat}\,W_1 + b_1)\,W_2 + b_2 \in \mathbb{R}^{d_{llm}}$$

        これで 16 個のパッチが、**LLM が「単語」と同じように扱える 16 個の画像トークン** になります。
        """
    )
    return


@app.cell
def _(gelu, np, patch_feats, rng):
    def build_projector(d_vision, d_hidden, d_llm, rng):
        return dict(
            W1=rng.standard_normal((d_vision, d_hidden)) / np.sqrt(d_vision),
            b1=np.zeros(d_hidden),
            W2=rng.standard_normal((d_hidden, d_llm)) / np.sqrt(d_hidden),
            b2=np.zeros(d_llm),
        )

    def project(feats, proj):
        return gelu(feats @ proj["W1"] + proj["b1"]) @ proj["W2"] + proj["b2"]

    d_llm = 32
    projector = build_projector(d_vision=24, d_hidden=48, d_llm=d_llm, rng=rng)
    image_tokens = project(patch_feats, projector)      # (16, 32)
    ("projected image tokens (LLM space):", image_tokens.shape)
    return d_llm, image_tokens


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. マルチモーダル系列の組み立て

        LLaVA のプロンプトは、テキストの中に画像の「置き場所」を作り、そこに画像トークン列を差し込む形です:

        ```
        [BOS] "USER:" <IMAGE:16トークン> "what is this? ASSISTANT:"  →  （ここから生成）
        ```

        テキストは語彙埋め込みでベクトル化し、画像トークン（projector の出力）と**同じ次元**なので、
        単純に **行方向に連結** できます。これがマルチモーダル系列。
        """
    )
    return


@app.cell
def _(d_llm, image_tokens, np, rng):
    # おもちゃの語彙（トークンID→単語）
    vocab = ["<bos>", "USER:", "what", "is", "this", "?", "ASSISTANT:",
             "a", "red", "circle", "and", "yellow", "square"]
    tok2id = {t: i for i, t in enumerate(vocab)}
    text_embed = rng.standard_normal((len(vocab), d_llm)) * 0.1  # 語彙埋め込み

    def embed_text(tokens):
        return text_embed[[tok2id[t] for t in tokens]]

    # 画像の前後のテキスト
    prefix = embed_text(["<bos>", "USER:"])          # 画像より前
    suffix = embed_text(["what", "is", "this", "?", "ASSISTANT:"])  # 画像より後

    # ★ マルチモーダル系列 = テキスト前 + 画像トークン + テキスト後
    sequence = np.concatenate([prefix, image_tokens, suffix], axis=0)

    # 各位置が「テキスト or 画像」かの記録（可視化・実装で重要）
    seg = (["text"] * len(prefix) + ["image"] * len(image_tokens)
           + ["text"] * len(suffix))
    seq_labels = (["<bos>", "USER:"] + [f"img{p}" for p in range(len(image_tokens))]
                  + ["what", "is", "this", "?", "ASST:"])
    ("sequence length:", sequence.shape[0], "= 2 text + 16 image + 5 text")
    return seg, seq_labels, sequence


@app.cell
def _(plt, seg, seq_labels):
    def _plot_segments():
        colors = ["tab:blue" if s == "text" else "tab:orange" for s in seg]
        fig, ax = plt.subplots(figsize=(10, 1.8))
        ax.bar(range(len(seg)), [1] * len(seg), color=colors, width=1.0,
               edgecolor="white")
        ax.set_xticks(range(len(seg)))
        ax.set_xticklabels(seq_labels, rotation=90, fontsize=7)
        ax.set_yticks([])
        ax.set_title("assembled multimodal sequence  (blue=text, orange=image tokens)")
        fig.tight_layout()
        return fig

    _plot_segments()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. LLM に流す — 生成のための因果 attention

        組み上げた系列を、02 で作った **因果 Transformer** にそのまま通します。
        ここで大切なのは:

        - **画像トークンもテキストトークンも、LLM から見れば区別なく「系列上のトークン」**
        - causal mask により、生成する側（後ろのテキスト）は **前にある画像トークンを見られる** →
          だから画像の内容を踏まえて答えを作れる

        最終位置の隠れ状態に「語彙への出力層」をかければ、次の単語の確率が出ます。
        """
    )
    return


@app.cell
def _(causal_mask, gelu, layer_norm, mha, np, rng):
    def init_block(d, n_heads, d_ff, rng):
        s = 1.0 / np.sqrt(d)
        return dict(
            Wq=rng.standard_normal((d, d)) * s, Wk=rng.standard_normal((d, d)) * s,
            Wv=rng.standard_normal((d, d)) * s, Wo=rng.standard_normal((d, d)) * s,
            g1=np.ones(d), b1=np.zeros(d), g2=np.ones(d), b2=np.zeros(d),
            W1=rng.standard_normal((d, d_ff)) * s, bf1=np.zeros(d_ff),
            W2=rng.standard_normal((d_ff, d)) / np.sqrt(d_ff), bf2=np.zeros(d),
        )

    def run_llm(seq, blocks, n_heads):
        n = seq.shape[0]
        mask = causal_mask(n)
        x = seq
        last_w = None
        for bi, p in enumerate(blocks):
            ln = layer_norm(x, p["g1"], p["b1"])
            if bi == len(blocks) - 1:
                a, last_w = mha(ln, p["Wq"], p["Wk"], p["Wv"], p["Wo"],
                                n_heads, mask, return_w=True)
            else:
                a = mha(ln, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask)
            x = x + a
            h = layer_norm(x, p["g2"], p["b2"])
            x = x + (gelu(h @ p["W1"] + p["bf1"]) @ p["W2"] + p["bf2"])
        return x, last_w

    return init_block, run_llm


@app.cell
def _(d_llm, init_block, rng, run_llm, sequence):
    llm_blocks = [init_block(d_llm, n_heads=4, d_ff=64, rng=rng) for _ in range(3)]
    llm_out, llm_attn = run_llm(sequence, llm_blocks, n_heads=4)
    ("LLM output hidden states:", llm_out.shape, "/ attn:", llm_attn.shape)
    return llm_attn, llm_out


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. 「答えの位置」は画像をどれだけ見ているか

        最終層の attention 行列で、**ASSISTANT: の位置（＝これから答えを書く場所）の行** を見ると、
        画像トークン（オレンジの範囲）にどれだけ注意を向けているかが分かります。
        VLM が「画像を根拠に答える」とは、まさにこの **テキスト位置 → 画像トークンへの attention** のことです。
        """
    )
    return


@app.cell
def _(llm_attn, np, plt, seg, seq_labels):
    def _plot_attn():
        attn = llm_attn.mean(axis=0)  # head 平均 (n,n)
        n = attn.shape[0]
        img_idx = [i for i, s in enumerate(seg) if s == "image"]
        asst_row = len(seg) - 1  # 最後のトークン（ASSISTANT:）

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        im = axes[0].imshow(attn, cmap="magma", vmin=0)
        # 画像トークン範囲を枠で示す
        axes[0].axvspan(img_idx[0] - .5, img_idx[-1] + .5, color="tab:orange", alpha=0.15)
        axes[0].set_title("last-layer causal attention\n(orange band = image tokens)")
        axes[0].set_xlabel("key position"); axes[0].set_ylabel("query position")
        fig.colorbar(im, ax=axes[0], fraction=0.046)

        axes[1].bar(range(n), attn[asst_row], color=[
            "tab:orange" if s == "image" else "tab:blue" for s in seg])
        axes[1].set_title("attention FROM 'ASSISTANT:' position\n(orange=image tokens it looks at)")
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(seq_labels, rotation=90, fontsize=6)
        axes[1].set_ylabel("attention weight")
        fig.tight_layout()
        return fig

    _plot_attn()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## まとめ & 本物への橋渡し

        | LLaVA の部品 | 役割 | 学習するか |
        |-------------|------|-----------|
        | 視覚エンコーダ (CLIP ViT) | 画像 → パッチ特徴 | 基本は**凍結** |
        | **Projector (MLP)** | 視覚特徴 → LLM 空間へ翻訳 | **ここを学習**（LLaVA の心臓） |
        | 語彙埋め込み + LLM | 系列を理解し次語を生成 | 段階的に微調整 |
        | 系列の連結 | 画像トークン ⧺ テキストトークン | 実装の工夫 |

        **VLM の本質**: 「画像を、LLM が読めるトークンに翻訳して、言語系列に差し込む」。
        たったこれだけで、01〜04 で作ってきた部品が **見て・考えて・答える** モデルに統合されました。

        ### でも、ここで問題が出てきます

        LLaVA 素朴版にはいくつか弱点があります。次章 06 で扱う **Qwen-VL** はこれらを解きました:

        1. **解像度が固定** — 画像を常に同じサイズ（例 224×224）に潰す。細かい文字や大きい画像で情報が失われる。
           → Qwen-VL の **naive dynamic resolution**（可変解像度）
        2. **位置情報がテキスト用のまま** — 画像は本来 2 次元なのに、トークン列にすると「1 列の順番」しか残らない。
           → Qwen-VL の **M-RoPE**（縦・横・時間を分けて位置を表す回転位置埋め込み）
        3. **画像トークンが多すぎる** — 高解像度だとパッチが爆発する。
           → Qwen-VL の **vision merger**（隣接パッチをまとめてトークン数を圧縮）

        **次章 06** で、これら Qwen-VL の中核技術を実装し、最後に **本物の Qwen3-VL**
        （元リポジトリ `qwen-image-search` が使っているモデル）へ橋渡しします。
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
