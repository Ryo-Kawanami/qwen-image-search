import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 03. Vision Transformer (ViT) — 画像を「トークンの列」にする

        ここが VLM への最初の大きな分岐点です。

        01〜02 で作った Transformer は「単語の列」を処理しました。
        **同じ機械に画像を食わせるには、画像を『トークンの列』に変えればいい** ——
        これが Vision Transformer (ViT, Dosovitskiy et al. 2020) の中心アイデアです。

        > 文 = 単語の列 → 画像 = **パッチの列**

        やることは 4 つ:

        1. **Patchify**: 画像を格子状の小パッチに切り、各パッチを 1 本のベクトルに潰す
        2. **Patch Embedding**: 線形層で `d_model` 次元に投影（= ViT の「単語埋め込み」）
        3. **[CLS] トークン + 位置埋め込み**: 全体を代表するトークンを先頭に足し、位置情報を注入
        4. あとは **02 の Transformer をそのまま** 通すだけ

        本章も NumPy。02 のブロックを部品として再利用します。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    return np, plt, rng


@app.cell
def _(np):
    # --- 01〜02 の部品を再掲（この章の土台） ---
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
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        split = lambda M: M.reshape(n, n_heads, dh).transpose(1, 0, 2)
        Qh, Kh, Vh = split(Q), split(K), split(V)
        outs, ws = [], []
        for h in range(n_heads):
            o, w = sdpa(Qh[h], Kh[h], Vh[h], mask)
            outs.append(o)
            ws.append(w)
        out = np.concatenate(outs, axis=-1) @ Wo
        return (out, np.stack(ws)) if return_w else out

    def layer_norm(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return g * (x - mu) / np.sqrt(var + eps) + b

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    return gelu, layer_norm, mha, softmax


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. 題材の画像を用意する

        オフラインで再現できるよう、NumPy で単純な合成画像を描きます（青い背景・赤い円・黄色い四角）。
        画像は `(H, W, 3)` の配列で、値は 0〜1。
        """
    )
    return


@app.cell
def _(np):
    def make_image(size=48):
        H = W = size
        img = np.zeros((H, W, 3))
        img[:, :, 2] = 0.3  # 全体をうっすら青背景に
        yy, xx = np.mgrid[0:H, 0:W]
        # 赤い円（中心やや上）
        circle = (xx - W * 0.45) ** 2 + (yy - H * 0.4) ** 2 < (size * 0.22) ** 2
        img[circle] = [0.9, 0.1, 0.1]
        # 黄色い四角（右下）
        sq = (xx > W * 0.62) & (xx < W * 0.9) & (yy > H * 0.62) & (yy < H * 0.9)
        img[sq] = [0.95, 0.9, 0.1]
        return img

    image = make_image(48)
    image.shape
    return (image,)


@app.cell
def _(image, plt):
    def _show():
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(image)
        ax.set_title("input image (48x48x3)")
        ax.axis("off")
        fig.tight_layout()
        return fig

    _show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Patchify — 画像を格子で切る

        パッチサイズ $P=8$ とすると、$48/8 = 6$ なので **6×6 = 36 パッチ** に分かれます。
        各パッチは $8\times8\times3 = 192$ 次元のベクトルになります。
        下の図で、切り分けたパッチを格子で確認します。
        """
    )
    return


@app.cell
def _(np):
    def patchify(img, P):
        """img (H,W,C) → patches (n_patches, P*P*C) と格子サイズ (gh, gw)."""
        H, W, C = img.shape
        gh, gw = H // P, W // P
        patches = []
        for i in range(gh):
            for j in range(gw):
                patch = img[i * P:(i + 1) * P, j * P:(j + 1) * P, :]
                patches.append(patch.reshape(-1))  # flatten → (P*P*C,)
        return np.stack(patches), (gh, gw)

    return (patchify,)


@app.cell
def _(image, patchify, plt):
    def _show_patches():
        P = 8
        patches, (gh, gw) = patchify(image, P)
        fig, axes = plt.subplots(gh, gw, figsize=(4.5, 4.5))
        for idx in range(gh * gw):
            ax = axes[idx // gw, idx % gw]
            ax.imshow(patches[idx].reshape(P, P, 3))
            ax.axis("off")
        fig.suptitle(f"{gh}x{gw} = {gh*gw} patches, each {P}x{P}x3 = {P*P*3}-dim")
        fig.tight_layout()
        return fig

    _show_patches()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        画像が 36 個の「視覚トークン」に分解されました。これは 02 の「8 個の単語トークン」と同じ形です。
        あとは言語とほぼ同じ流れ。

        ## 3. Patch Embedding + [CLS] + 位置埋め込み

        - **Patch Embedding**: 各パッチ（192次元）を線形層で `d_model` 次元へ。これが ViT の「単語埋め込み」。
        - **[CLS] トークン**: 学習可能なベクトルを先頭に足す。全パッチから情報を集約し、最後に画像全体の代表として使う。
        - **位置埋め込み**: パッチの並び順（どこのパッチか）を教える。ViT では sin/cos ではなく **学習型** が一般的。
        """
    )
    return


@app.cell
def _(np, patchify, rng):
    def build_tokens(img, P, d_model, rng):
        patches, grid = patchify(img, P)           # (n_patch, patch_dim)
        patch_dim = patches.shape[1]

        # patch embedding（本来は学習で決まる線形層）
        W_embed = rng.standard_normal((patch_dim, d_model)) / np.sqrt(patch_dim)
        tokens = patches @ W_embed                  # (n_patch, d_model)

        # [CLS] トークンを先頭に連結
        cls = rng.standard_normal((1, d_model)) * 0.02
        tokens = np.concatenate([cls, tokens], axis=0)  # (1+n_patch, d_model)

        # 学習型の位置埋め込み（ランダム初期化で代用）
        pos = rng.standard_normal(tokens.shape) * 0.02
        tokens = tokens + pos
        return tokens, grid

    _t, _g = build_tokens(np.zeros((48, 48, 3)), 8, 32, rng)
    ("sanity check shapes:", _t.shape, _g)
    return (build_tokens,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Transformer を通す（02 のブロックそのまま）

        画像用に特別なことは **何もありません**。02 で作った Pre-LN Transformer ブロックをそのまま積みます。
        違いは「causal mask を使わない」点だけ。画像のパッチは時系列ではないので、**全パッチが互いを見て良い**（双方向）。
        """
    )
    return


@app.cell
def _(gelu, layer_norm, mha, np, rng):
    def init_block(d_model, n_heads, d_ff, rng):
        s = 1.0 / np.sqrt(d_model)
        return dict(
            Wq=rng.standard_normal((d_model, d_model)) * s,
            Wk=rng.standard_normal((d_model, d_model)) * s,
            Wv=rng.standard_normal((d_model, d_model)) * s,
            Wo=rng.standard_normal((d_model, d_model)) * s,
            g1=np.ones(d_model), b1=np.zeros(d_model),
            g2=np.ones(d_model), b2=np.zeros(d_model),
            W1=rng.standard_normal((d_model, d_ff)) * s, bf1=np.zeros(d_ff),
            W2=rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff), bf2=np.zeros(d_model),
        )

    def block(x, p, n_heads, return_w=False):
        # マスクなし = 双方向 self-attention
        a = mha(layer_norm(x, p["g1"], p["b1"]),
                p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, return_w=return_w)
        if return_w:
            a, w = a
        x = x + a
        h = layer_norm(x, p["g2"], p["b2"])
        f = gelu(h @ p["W1"] + p["bf1"]) @ p["W2"] + p["bf2"]
        x = x + f
        return (x, w) if return_w else x

    return block, init_block


@app.cell
def _(block, build_tokens, image, init_block, rng):
    # ViT ミニチュアを構築して forward
    d_model, n_heads, d_ff, n_layers = 32, 4, 64, 3
    vit_tokens, vit_grid = build_tokens(image, P=8, d_model=d_model, rng=rng)
    vit_blocks = [init_block(d_model, n_heads, d_ff, rng) for _ in range(n_layers)]

    h_vit = vit_tokens
    for _bi, _p in enumerate(vit_blocks):
        # 最終層だけ attention 重みも取り出す
        if _bi == n_layers - 1:
            h_vit, last_w = block(h_vit, _p, n_heads, return_w=True)
        else:
            h_vit = block(h_vit, _p, n_heads)

    cls_out = h_vit[0]  # [CLS] の出力 = 画像全体の表現
    ("hidden shape", h_vit.shape, "/ CLS vector shape", cls_out.shape)
    return last_w, n_heads, vit_grid


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. [CLS] は画像のどこを見ているか

        最終層で **[CLS] トークンが各パッチへ向ける attention 重み** を取り出し、
        元の 6×6 格子に戻して画像に重ねます。これは実際の ViT 解釈でよく使う可視化です
        （head 平均を取っています。乱数重みなので「意味」はまだありませんが、仕組みは本物と同じ）。
        """
    )
    return


@app.cell
def _(image, last_w, n_heads, np, plt, vit_grid):
    def _plot_cls_attention():
        gh, gw = vit_grid
        # last_w: (n_heads, seq, seq)。行0 = CLS が各トークンへ向ける重み。列0(=CLS自身)を除く
        cls_attn = last_w[:, 0, 1:].mean(axis=0)     # head 平均 → (n_patch,)
        attn_map = cls_attn.reshape(gh, gw)
        attn_up = np.kron(attn_map, np.ones((8, 8)))  # 48x48 に拡大

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        axes[0].imshow(image); axes[0].set_title("image"); axes[0].axis("off")
        axes[1].imshow(attn_map, cmap="inferno")
        axes[1].set_title("CLS attention\n(6x6 patch grid)"); axes[1].axis("off")
        axes[2].imshow(image)
        axes[2].imshow(attn_up, cmap="inferno", alpha=0.55)
        axes[2].set_title("overlay"); axes[2].axis("off")
        fig.tight_layout()
        return fig

    _plot_cls_attention()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## まとめ & 本物への橋渡し

        | ViT の部品 | 対応する言語版（02） | PyTorch / HF |
        |------------|--------------------|-------------|
        | patchify + patch embedding | 単語埋め込み | `nn.Conv2d(kernel=stride=P)` 一発で実装される |
        | [CLS] トークン | （BERT の [CLS] と同じ） | 学習パラメータ |
        | 学習型 位置埋め込み | sin/cos 位置エンコーディング | `nn.Parameter` |
        | 双方向 Transformer ブロック | causal ブロック | `ViTModel`（transformers） |

        **重要な気づき**: 画像専用の魔法はほとんどありません。**「画像をトークン列にする入口」さえ作れば、
        あとは言語と同じ Transformer**。この普遍性こそが、次章以降で画像と言語を同じ土俵に載せられる理由です。

        > 実務では、patch embedding は `Conv2d(in=3, out=d_model, kernel=P, stride=P)` 一発で書けます
        > （パッチ切り出しと線形投影が畳み込み 1 回に一致するため）。CLIP や Qwen-VL の画像側もこの形です。

        **次章 04（CLIP）** では、この ViT（画像エンコーダ）と 02 の Transformer（テキストエンコーダ）を
        **並べて**、「画像とテキストを同じベクトル空間に置く」対照学習を実装します。
        VLM が画像と言葉を結びつける、その最初の接着剤です。
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
