import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 02. Transformer — Attention を「ブロック」に組み上げる

        前章の Multi-Head Attention は強力ですが、それ単体ではモデルになりません。
        Transformer は attention の周りに次の部品を足して、積み重ね可能な **ブロック** にします。

        1. **位置エンコーディング（Positional Encoding）** — attention は順序を知らないので、位置情報を注入する
        2. **残差接続（Residual / skip connection）** — 深く積んでも学習が壊れないようにする
        3. **Layer Normalization** — 各層の出力を整えて安定化
        4. **Feed-Forward Network（FFN）** — トークンごとに非線形変換して表現力を上げる

        これらを組むと、GPT や BERT、そして VLM の「言語側」の背骨になります。
        本章も NumPy でゼロから。前章の attention をそのまま部品として使います。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    return np, plt, rng


@app.cell
def _(np):
    # --- 前章 01 で作った attention を再掲（この章の部品として使う） ---
    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)
        weights = softmax(scores, axis=-1)
        return weights @ V, weights

    def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None):
        n, d_model = X.shape
        d_head = d_model // n_heads
        Q, K, V = X @ W_q, X @ W_k, X @ W_v

        def split(M):
            return M.reshape(n, n_heads, d_head).transpose(1, 0, 2)

        Qh, Kh, Vh = split(Q), split(K), split(V)
        outs = [scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask)[0]
                for h in range(n_heads)]
        return np.concatenate(outs, axis=-1) @ W_o

    return multi_head_attention, softmax


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. 位置エンコーディング — 順序をどう教えるか

        Attention は集合演算です。トークンを並べ替えても、重みの付き方は同じ（順序に無頓着）。
        しかし言語では「犬が猫を追う」と「猫が犬を追う」は別物。**位置情報が必要** です。

        オリジナルの Transformer は、位置 $pos$ と次元 $i$ に対して sin/cos を使います:

        $$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\quad
          PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

        波長を次元ごとに変えることで、各位置に固有の「指紋」を与えます。
        これを埋め込みに **足し算** します。
        """
    )
    return


@app.cell
def _(np):
    def sinusoidal_positional_encoding(seq_len, d_model):
        pos = np.arange(seq_len)[:, None]          # (seq_len, 1)
        i = np.arange(d_model)[None, :]            # (1, d_model)
        angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angle[:, 0::2])       # 偶数次元は sin
        pe[:, 1::2] = np.cos(angle[:, 1::2])       # 奇数次元は cos
        return pe

    return (sinusoidal_positional_encoding,)


@app.cell
def _(plt, sinusoidal_positional_encoding):
    def _plot_pe():
        pe = sinusoidal_positional_encoding(seq_len=50, d_model=64)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(pe, aspect="auto", cmap="RdBu")
        ax.set_xlabel("dimension")
        ax.set_ylabel("position")
        ax.set_title("sinusoidal positional encoding\neach row = a unique fingerprint for a position")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        return fig

    _plot_pe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        各行（位置）が違う縞模様を持ち、位置の「指紋」になっているのが分かります。

        > **ここが伏線です。** この「位置をどう表すか」という問題は、後で **RoPE（回転式位置埋め込み）**、
        > さらに Qwen-VL の **M-RoPE（画像の縦・横・時間の位置を扱う）** へと進化します（06章）。
        > いまは「足し算で注入する sin/cos 版」を押さえておきましょう。

        ## 2. Layer Normalization

        各トークンのベクトルを、その特徴次元方向に平均 0・分散 1 へ正規化し、
        学習可能な $\gamma$（スケール）と $\beta$（シフト）で調整します。層を深くしても値が発散しないための安定剤です。
        """
    )
    return


@app.cell
def _(np):
    def layer_norm(x, gamma, beta, eps=1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + eps)
        return gamma * x_hat + beta

    return (layer_norm,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Feed-Forward Network（FFN）

        Attention が「トークン間で情報を混ぜる」役なら、FFN は「各トークンを個別に深く変換する」役です。
        中間で一度大きく広げて（通常 4 倍）、非線形（GELU）を通し、また戻します。

        $$\mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1)\,W_2 + b_2$$
        """
    )
    return


@app.cell
def _(np):
    def gelu(x):
        # tanh 近似（GPT などで使われる形）
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def feed_forward(x, W1, b1, W2, b2):
        return gelu(x @ W1 + b1) @ W2 + b2

    return feed_forward, gelu


@app.cell
def _(gelu, np, plt):
    def _plot_gelu():
        x = np.linspace(-4, 4, 200)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, gelu(x), label="GELU")
        ax.plot(x, np.maximum(0, x), "--", alpha=0.6, label="ReLU")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.legend()
        ax.set_title("GELU vs ReLU")
        fig.tight_layout()
        return fig

    _plot_gelu()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. 残差接続と Transformer ブロック

        部品が揃いました。組み立てます。現代的な **Pre-LN**（正規化を先に置く）構成を使います:

        ```
        x = x + MHA(LayerNorm(x))     # attention で混ぜて、元に足し戻す
        x = x + FFN(LayerNorm(x))     # 各トークンを変換して、また足し戻す
        ```

        `x + ...` の **足し戻し（残差）** が肝です。これにより、そのブロックが何もしなくても
        情報がそのまま素通りでき、深く積んでも勾配が消えません。
        """
    )
    return


@app.cell
def _(feed_forward, layer_norm, multi_head_attention, np, rng):
    def init_block_params(d_model, n_heads, d_ff, rng):
        """1ブロック分の重みをランダム初期化して dict で返す。"""
        s = 1.0 / np.sqrt(d_model)
        return dict(
            Wq=rng.standard_normal((d_model, d_model)) * s,
            Wk=rng.standard_normal((d_model, d_model)) * s,
            Wv=rng.standard_normal((d_model, d_model)) * s,
            Wo=rng.standard_normal((d_model, d_model)) * s,
            g1=np.ones(d_model), b1=np.zeros(d_model),   # LN(attn 前)
            g2=np.ones(d_model), b2=np.zeros(d_model),   # LN(ffn 前)
            W1=rng.standard_normal((d_model, d_ff)) * s,
            bf1=np.zeros(d_ff),
            W2=rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff),
            bf2=np.zeros(d_model),
        )

    def transformer_block(x, p, n_heads, mask=None):
        # --- sub-layer 1: multi-head self-attention（Pre-LN + 残差） ---
        a = multi_head_attention(
            layer_norm(x, p["g1"], p["b1"]),
            p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask,
        )
        x = x + a
        # --- sub-layer 2: FFN（Pre-LN + 残差） ---
        f = feed_forward(
            layer_norm(x, p["g2"], p["b2"]),
            p["W1"], p["bf1"], p["W2"], p["bf2"],
        )
        x = x + f
        return x

    return init_block_params, transformer_block


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. ブロックを積む = Transformer 本体

        埋め込み + 位置エンコーディングを入力に、ブロックを $L$ 段重ねます。
        ここでは「トークン ID の列」を語彙埋め込みしてから流す、GPT ミニチュアを作ります。
        """
    )
    return


@app.cell
def _(
    init_block_params,
    np,
    rng,
    sinusoidal_positional_encoding,
    transformer_block,
):
    def causal_mask(n):
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    class TinyGPT:
        def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, rng):
            self.d_model = d_model
            self.n_heads = n_heads
            self.embed = rng.standard_normal((vocab_size, d_model)) * 0.1
            self.blocks = [init_block_params(d_model, n_heads, d_ff, rng)
                           for _ in range(n_layers)]

        def forward(self, token_ids):
            n = len(token_ids)
            x = self.embed[token_ids]                                   # (n, d_model)
            x = x + sinusoidal_positional_encoding(n, self.d_model)     # 位置を注入
            mask = causal_mask(n)                                       # 未来を隠す
            for p in self.blocks:
                x = transformer_block(x, p, self.n_heads, mask)
            return x

    # おもちゃGPT: 語彙20, d_model=32, 4 head, FFN 64, 3層
    tiny = TinyGPT(vocab_size=20, d_model=32, n_heads=4, d_ff=64, n_layers=3, rng=rng)
    tokens_demo = [3, 1, 4, 1, 5, 9, 2, 6]
    hidden = tiny.forward(tokens_demo)
    ("入力トークン列", tokens_demo, "→ 出力の隠れ状態 shape", hidden.shape)
    return hidden, tokens_demo


@app.cell
def _(mo):
    mo.md(
        r"""
        各トークンが `(位置・文脈込みの) d_model 次元ベクトル` に変換されました。
        この隠れ状態の上に「次トークン予測」の線形層を載せれば GPT、`[CLS]` を載せれば分類器…と応用します。

        下は 8 トークンの出力隠れ状態を可視化したもの。**因果マスクのおかげで、後ろのトークンほど
        多くの文脈を取り込んで表現が変化していく**様子が（乱数重みでも）見て取れます。
        """
    )
    return


@app.cell
def _(hidden, plt, tokens_demo):
    def _plot_hidden():
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(hidden, aspect="auto", cmap="coolwarm")
        ax.set_yticks(range(len(tokens_demo)))
        ax.set_yticklabels([f"tok {t}" for t in tokens_demo])
        ax.set_xlabel("hidden dimension")
        ax.set_title("output hidden states of TinyGPT (per token)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        return fig

    _plot_hidden()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## まとめ & 本物への橋渡し

        | 部品 | 役割 | 対応する PyTorch |
        |------|------|-----------------|
        | `sinusoidal_positional_encoding` | 順序の注入 | `nn.Embedding`（学習型）や RoPE |
        | `layer_norm` | 層の安定化 | `nn.LayerNorm` |
        | `feed_forward` + `gelu` | トークン毎の非線形変換 | `nn.Linear`×2 + `nn.GELU` |
        | `transformer_block` | 混ぜる→変換する 1 段 | `nn.TransformerEncoderLayer` |
        | `TinyGPT` | ブロックを積んだ本体 | `nn.TransformerEncoder` / GPT |

        Attention（01）→ ブロック化（02）で、**言語モデルの背骨**ができました。

        **ここからが VLM への分岐点です。** 次章 03 では、この Transformer を **画像** に適用します。
        「文＝単語の列」だったものを「画像＝パッチの列」と読み替えるだけで、
        まったく同じ Transformer が画像を理解し始めます。それが **Vision Transformer (ViT)** です。
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
