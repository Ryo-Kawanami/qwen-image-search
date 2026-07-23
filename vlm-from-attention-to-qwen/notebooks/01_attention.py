import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 01. Attention — すべての出発点

        VLM への長い旅は、たった一つのアイデアから始まります。**Attention（注意機構）** です。

        > 「今この単語（や画像パッチ）を理解するために、他のどこに注目すべきか？」

        Transformer も、ViT も、CLIP も、Qwen-VL も、中核はこの Attention です。
        ここでは **NumPy だけ** で attention をゼロから組み立て、行列演算のレベルで「何が起きているか」を見ます。

        この章のゴール:

        1. **Scaled Dot-Product Attention** を式のまま実装する
        2. **Self-Attention**（自分自身に注目する）を理解する
        3. **Causal Mask**（未来を見せない）を入れる
        4. **Multi-Head Attention**（複数の視点で見る）に拡張する
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    return np, plt, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Query・Key・Value という比喩

        Attention は「辞書引き」に例えられます。

        - **Query（クエリ）**: 「私は何を探しているか」
        - **Key（キー）**: 各要素の「見出し」
        - **Value（バリュー）**: 各要素が持つ「中身」

        Query と各 Key の**相性（内積）**を測り、相性の高い要素の Value を強く混ぜて取り出します。
        「相性」を確率（重み）に変えるのが softmax です。

        式で書くとこれだけです:

        $$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

        一つずつ組み立てましょう。まずは softmax。
        """
    )
    return


@app.cell
def _(np):
    def softmax(x, axis=-1):
        """数値的に安定な softmax。最大値を引いてから exp する。"""
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    return (softmax,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### なぜ $\sqrt{d_k}$ で割るのか

        $Q$ と $K$ の次元 $d_k$ が大きいと、内積 $QK^\top$ の値が大きくなりがちです。
        大きすぎる値を softmax に入れると、ほぼ 0/1 の「尖った」分布になり、勾配が消えて学習が進みません。
        $\sqrt{d_k}$ で割ると分散が概ね一定に保たれ、分布が安定します。下でその効果を確認します。
        """
    )
    return


@app.cell
def _(np, plt, rng, softmax):
    def _demo_scaling():
        d_k = 64
        q = rng.standard_normal(d_k)
        k = rng.standard_normal((6, d_k))
        raw = k @ q  # スケーリングなしの内積
        scaled = raw / np.sqrt(d_k)  # sqrt(d_k) で割る

        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].bar(range(6), softmax(raw))
        axes[0].set_title(f"no scaling (max={raw.max():.1f})\ntoo peaky")
        axes[0].set_ylim(0, 1)
        axes[1].bar(range(6), softmax(scaled), color="tab:green")
        axes[1].set_title(f"divide by sqrt(d_k) (max={scaled.max():.1f})\nsmooth")
        axes[1].set_ylim(0, 1)
        fig.tight_layout()
        return fig

    _demo_scaling()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Scaled Dot-Product Attention

        式をそのままコードにします。`mask` は後で使う「見てはいけない場所」を $-\infty$ にするための引数です。
        """
    )
    return


@app.cell
def _(np, softmax):
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Q: (n_q, d_k)   探す側
        K: (n_k, d_k)   見出し
        V: (n_k, d_v)   中身
        mask: (n_q, n_k) の bool。True の位置は「見てはいけない」→ -inf にする
        戻り値: out (n_q, d_v), weights (n_q, n_k)
        """
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)  # (n_q, n_k) 相性スコア
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)
        weights = softmax(scores, axis=-1)  # 各 query が n_k 個へ配分する確率
        out = weights @ V  # 確率で Value を加重平均
        return out, weights

    return (scaled_dot_product_attention,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 具体例で動かす

        5 個のトークンからなる系列を考えます。各トークンを 4 次元のベクトルで表します（`d_k = d_v = 4`）。
        ここではまだ「学習された重み」はなく、Q=K=V=入力そのもの、として**素の self-attention** を見ます。
        """
    )
    return


@app.cell
def _(np, scaled_dot_product_attention):
    # 5トークン × 4次元。わざと似たベクトルのペアを作る（0番と3番、1番と4番が似ている）
    X_demo = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # token 0
            [0.0, 1.0, 0.0, 0.0],  # token 1
            [0.0, 0.0, 1.0, 0.0],  # token 2
            [0.9, 0.1, 0.0, 0.0],  # token 3  ← 0 に似ている
            [0.0, 0.9, 0.1, 0.0],  # token 4  ← 1 に似ている
        ]
    )
    out_demo, w_demo = scaled_dot_product_attention(X_demo, X_demo, X_demo)
    w_demo.round(2)
    return X_demo, w_demo


@app.cell
def _(mo):
    mo.md(
        r"""
        上の行列 `w_demo` の各行が「そのトークンが他のどのトークンをどれだけ見ているか」です。
        行の合計は 1（確率）。ヒートマップで見ると、**似たトークン同士が強く注目し合っている**のが分かります。
        """
    )
    return


@app.cell
def _(plt, w_demo):
    def _plot_weights():
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(w_demo, cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("key (attended to)")
        ax.set_ylabel("query (attending)")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{w_demo[i, j]:.2f}", ha="center", va="center",
                        color="white" if w_demo[i, j] < 0.5 else "black", fontsize=8)
        ax.set_title("attention weights\n(0<->3, 1<->4 attend to each other)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig

    _plot_weights()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. 学習される投影（projection）

        実際の attention では、入力 $X$ をそのまま Q/K/V にはしません。
        **学習可能な重み行列** $W_Q, W_K, W_V$ で別々の空間に投影します:

        $$Q = X W_Q,\quad K = X W_K,\quad V = X W_V$$

        これにより「探すための表現」「見出しのための表現」「中身のための表現」を役割分担できます。
        ここでは学習の代わりにランダムな重みを置いて、shape の流れだけ確認します。
        """
    )
    return


@app.cell
def _(np, rng, scaled_dot_product_attention):
    def self_attention(X, W_q, W_k, W_v, mask=None):
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        return scaled_dot_product_attention(Q, K, V, mask=mask)

    def make_projections(d_model, d_k, d_v, rng):
        """ランダム初期化した投影行列（本来は学習で決まる）。"""
        scale = 1.0 / np.sqrt(d_model)
        W_q = rng.standard_normal((d_model, d_k)) * scale
        W_k = rng.standard_normal((d_model, d_k)) * scale
        W_v = rng.standard_normal((d_model, d_v)) * scale
        return W_q, W_k, W_v

    # 6トークン × d_model=8 の系列で試す
    X_proj = rng.standard_normal((6, 8))
    Wq, Wk, Wv = make_projections(d_model=8, d_k=8, d_v=8, rng=rng)
    out_proj, w_proj = self_attention(X_proj, Wq, Wk, Wv)
    (out_proj.shape, w_proj.shape)
    return (self_attention,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Causal Mask — 未来を見せない

        言語モデル（GPT 系）は「次の単語」を予測します。
        学習時に $i$ 番目のトークンが $i+1, i+2, \dots$（＝答え）を見てしまうとカンニングになります。
        そこで **上三角部分を $-\infty$** にして、各トークンが「自分より前」しか見られないようにします。
        これが **causal（因果的）mask**。GPT と、後で出てくる VLM の「言語側」で使われます。
        """
    )
    return


@app.cell
def _(np, plt, rng, self_attention):
    def causal_mask(n):
        """(n, n) の bool。True の位置（未来）は見えない。"""
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    def _demo_causal():
        n = 6
        X = rng.standard_normal((n, 4))
        Wq = rng.standard_normal((4, 4)) * 0.5
        Wk = rng.standard_normal((4, 4)) * 0.5
        Wv = rng.standard_normal((4, 4)) * 0.5
        _, w = self_attention(X, Wq, Wk, Wv, mask=causal_mask(n))

        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(w, cmap="magma", vmin=0, vmax=1)
        ax.set_title("causal attention\nweight only on lower triangle")
        ax.set_xlabel("key (attended to)")
        ax.set_ylabel("query (attending)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig

    _demo_causal()
    return (causal_mask,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ヒートマップの**右上（未来）が真っ黒＝重み 0** になっています。
        0 番目のトークンは自分しか見えず、5 番目は全員を見られる、という三角形です。

        ## 5. Multi-Head Attention — 複数の視点

        1 組の $Q,K,V$ だと「一種類の注目の仕方」しかできません。
        実際には「文法的な関係」「意味的な関係」など複数の観点で同時に見たい。
        そこで表現を $h$ 個の **head** に分割し、それぞれ独立に attention してから連結します。

        $$\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
        $$\mathrm{MHA} = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O$$

        各 head の次元は $d_{model}/h$。連結すると元の次元に戻り、最後に $W^O$ で混ぜます。
        """
    )
    return


@app.cell
def _(np, scaled_dot_product_attention):
    def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None):
        """
        X: (n, d_model)
        W_q,W_k,W_v: (d_model, d_model)   （全 head 分をまとめた投影）
        W_o: (d_model, d_model)           出力の混合
        n_heads: head の数。d_model は n_heads で割り切れること。
        """
        n, d_model = X.shape
        d_head = d_model // n_heads

        Q = X @ W_q  # (n, d_model)
        K = X @ W_k
        V = X @ W_v

        # (n, d_model) を (n_heads, n, d_head) に分割
        def split_heads(M):
            return M.reshape(n, n_heads, d_head).transpose(1, 0, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        head_outs = []
        head_weights = []
        for h in range(n_heads):
            o, w = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask=mask)
            head_outs.append(o)          # (n, d_head)
            head_weights.append(w)       # (n, n)

        # 連結して (n, d_model) に戻し、W_o で混ぜる
        concat = np.concatenate(head_outs, axis=-1)  # (n, d_model)
        out = concat @ W_o
        return out, np.stack(head_weights)  # weights: (n_heads, n, n)

    return (multi_head_attention,)


@app.cell
def _(causal_mask, multi_head_attention, np, rng):
    # 8トークン × d_model=16、4 head（各 head 4 次元）で動かす
    def _run_mha():
        n, d_model, n_heads = 8, 16, 4
        X = rng.standard_normal((n, d_model))
        mk = lambda: rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        out, weights = multi_head_attention(
            X, mk(), mk(), mk(), mk(), n_heads=n_heads, mask=causal_mask(n)
        )
        return out.shape, weights.shape

    _run_mha()
    return


@app.cell
def _(causal_mask, multi_head_attention, np, plt, rng):
    def _plot_heads():
        n, d_model, n_heads = 8, 16, 4
        X = rng.standard_normal((n, d_model))
        mk = lambda: rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        _, weights = multi_head_attention(
            X, mk(), mk(), mk(), mk(), n_heads=n_heads, mask=causal_mask(n)
        )
        fig, axes = plt.subplots(1, n_heads, figsize=(11, 3))
        for h in range(n_heads):
            axes[h].imshow(weights[h], cmap="cividis", vmin=0, vmax=1)
            axes[h].set_title(f"head {h}")
            axes[h].set_xticks([])
            axes[h].set_yticks([])
        fig.suptitle("each head can learn a different attention pattern (random here)")
        fig.tight_layout()
        return fig

    _plot_heads()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        4 つの head がそれぞれ別のパターンで注目しています。学習後は、ある head は「直前の単語」、
        別の head は「文頭の主語」…のように役割分担が生まれます。

        ---

        ## まとめ & 本物への橋渡し

        今日ゼロから作ったもの:

        | 作ったもの | 役割 |
        |-----------|------|
        | `softmax` + `√d_k` | 相性スコアを安定した確率へ |
        | `scaled_dot_product_attention` | attention の核。**この 1 関数が全ての土台** |
        | `self_attention` | $W_Q,W_K,W_V$ で投影してから attention |
        | `causal_mask` | 言語モデルで未来を隠す |
        | `multi_head_attention` | 複数視点で同時に注目 |

        **PyTorch では** これは `torch.nn.MultiheadAttention` や、`F.scaled_dot_product_attention`
        （FlashAttention 実装が裏で走る）に相当します。私たちが `for h in range(n_heads)` で回した部分は、
        実際にはバッチ次元と head 次元をまとめた 4 次元テンソルの一括行列積で高速化されています。

        **次章（02 Transformer）** では、この Multi-Head Attention を部品として、
        LayerNorm・FFN・残差接続と組み合わせ、言語モデルの基本ブロック **Transformer** を完成させます。
        そこから画像（03 ViT）へと同じ部品を持ち込んでいきます。
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
