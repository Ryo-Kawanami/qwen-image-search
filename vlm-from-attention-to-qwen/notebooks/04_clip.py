import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 04. CLIP — 画像とテキストを「同じ空間」に並べる

        03 で画像を、02 でテキストを、それぞれベクトルに変換できるようになりました。
        でも両者は **バラバラの空間** にいます。犬の画像ベクトルと "a dog" のベクトルは、まだ無関係。

        **CLIP (Radford et al. 2021)** のアイデアはシンプルです:

        > 対応する画像とテキストのベクトルが **近く** なり、対応しないペアが **遠く** なるように学習する

        これを **対照学習（contrastive learning）** と呼びます。
        画像エンコーダ（ViT, 03）とテキストエンコーダ（Transformer, 02）を**別々に**持ち、
        出力を一つの共有空間に射影して、コサイン類似度で結びつけます。

        この「画像とテキストを同じ物差しで測れる」性質が、

        - **ゼロショット画像分類**（学習していないラベルでも分類できる）
        - **画像検索**（← 元リポジトリ `qwen-image-search` がまさにこれ！）
        - そして **VLM の視覚エンコーダの初期化**

        を可能にします。本章では対照学習の中身を NumPy で組み、損失が何を引き寄せ何を引き離すのかを見ます。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    return np, plt, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. デュアルエンコーダの構造

        ```
          画像 ──[ ViT: 画像エンコーダ ]──▶ 画像ベクトル ──[線形射影]──▶  ┐
                                                                          ├─ 同じ d 次元の共有空間
          テキスト ─[ Transformer: テキストエンコーダ ]─▶ テキストベクトル ─[線形射影]──▶ ┘
        ```

        2 本のエンコーダは 02・03 で作ったものです。ここでは中身を再実装せず、
        「エンコーダは何らかのベクトルを出す箱」として抽象化し、**対照学習のコア** に集中します。
        まずは L2 正規化（ベクトルを長さ 1 にする）と、コサイン類似度から。
        """
    )
    return


@app.cell
def _(np):
    def l2_normalize(x, axis=-1, eps=1e-8):
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

    def cosine_sim_matrix(A, B):
        """A(n,d), B(m,d) の全ペアのコサイン類似度 (n,m)。"""
        return l2_normalize(A) @ l2_normalize(B).T

    return cosine_sim_matrix, l2_normalize


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. 対照損失（InfoNCE / CLIP loss）

        $N$ 組の (画像, テキスト) ミニバッチを考えます。正規化した画像・テキスト特徴から
        類似度行列 $S \in \mathbb{R}^{N\times N}$ を作ります（$S_{ij}$ = 画像 $i$ と テキスト $j$ の類似度）。

        **対角成分 $S_{ii}$ が正しいペア**。これを温度 $\tau$ でスケールし、
        「各画像行は正しいテキスト列を選ぶ分類問題」「各テキスト列は正しい画像行を選ぶ分類問題」の
        **両方向の交差エントロピー** を平均します。

        $$\mathcal{L} = \tfrac{1}{2}\big(\mathrm{CE}(\text{行方向}, \text{正解}=対角) + \mathrm{CE}(\text{列方向}, \text{正解}=対角)\big)$$

        正解ラベルは常に `[0, 1, 2, ..., N-1]`（$i$ 番目の画像の正解は $i$ 番目のテキスト）。
        """
    )
    return


@app.cell
def _(cosine_sim_matrix, np):
    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def clip_loss(image_feats, text_feats, temperature=0.07):
        n = image_feats.shape[0]
        logits = cosine_sim_matrix(image_feats, text_feats) / temperature  # (n,n)
        labels = np.arange(n)  # 正解は対角

        # 行方向: 各画像 → 正しいテキスト
        p_i2t = softmax(logits, axis=1)
        loss_i2t = -np.mean(np.log(p_i2t[labels, labels] + 1e-9))
        # 列方向: 各テキスト → 正しい画像
        p_t2i = softmax(logits, axis=0)
        loss_t2i = -np.mean(np.log(p_t2i[labels, labels] + 1e-9))

        return 0.5 * (loss_i2t + loss_t2i), logits

    return clip_loss, softmax


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. 「揃っている」時と「バラバラ」の時で損失を比べる

        - **揃った特徴**: 画像 $i$ とテキスト $i$ をわざと似せたベクトルにする → 対角が強い → 損失は小さいはず
        - **ランダムな特徴**: 無関係 → 対角が目立たない → 損失は大きいはず

        4 ペア（例えば「犬・猫・車・花」の画像とその説明文）を想定して数値で確認します。
        """
    )
    return


@app.cell
def _(clip_loss, np, rng):
    def make_aligned(n, d, rng, noise=0.15):
        """正しいペアが似ている特徴を作る（学習が成功した後を模擬）。"""
        shared = rng.standard_normal((n, d))          # ペアごとの「意味」
        img = shared + noise * rng.standard_normal((n, d))
        txt = shared + noise * rng.standard_normal((n, d))
        return img, txt

    def make_random(n, d, rng):
        return rng.standard_normal((n, d)), rng.standard_normal((n, d))

    n_pairs, dim = 4, 16
    img_a, txt_a = make_aligned(n_pairs, dim, rng)
    img_r, txt_r = make_random(n_pairs, dim, rng)

    loss_aligned, logits_aligned = clip_loss(img_a, txt_a)
    loss_random, logits_random = clip_loss(img_r, txt_r)

    (
        f"aligned features  -> loss = {loss_aligned:.3f}",
        f"random features   -> loss = {loss_random:.3f}",
    )
    return logits_aligned, logits_random, n_pairs


@app.cell
def _(mo):
    mo.md(
        r"""
        揃った特徴の方が損失が小さいはずです。類似度行列（logits）を並べて見ると一目瞭然:
        **学習が成功すると対角線が光り、他は暗くなる**。これが「正しいペアだけ引き寄せる」ということです。
        """
    )
    return


@app.cell
def _(logits_aligned, logits_random, n_pairs, plt):
    def _plot():
        labels = ["dog", "cat", "car", "flower"][:n_pairs]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
        for ax, logits, title in [
            (axes[0], logits_random, "random (before training)"),
            (axes[1], logits_aligned, "aligned (after training)"),
        ]:
            im = ax.imshow(logits, cmap="viridis")
            ax.set_title(title)
            ax.set_xticks(range(n_pairs)); ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(range(n_pairs)); ax.set_yticklabels(labels)
            ax.set_xlabel("text"); ax.set_ylabel("image")
            for i in range(n_pairs):
                ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="red", lw=2))
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("similarity logits: diagonal (red) = correct pairs")
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. ミニ学習ループで対角を「光らせる」

        本当に勾配で学習できることを、超小型の例で示します。
        固定の「意味ベクトル」から作った画像特徴・テキスト特徴に、それぞれ学習可能な **射影行列** をかけ、
        clip_loss を数値微分で最小化します（教材用に単純な有限差分の勾配降下）。
        損失が下がり、対角がくっきりしていく様子を追います。
        """
    )
    return


@app.cell
def _(clip_loss, np, rng):
    def train_projection(steps=60, lr=0.5, n=4, d=12, seed=7):
        r = np.random.default_rng(seed)
        # 正しいペアは「意味」を共有するが、観測特徴は別の歪んだ空間にある
        meaning = r.standard_normal((n, d))
        img_raw = meaning @ r.standard_normal((d, d))   # 画像側の歪み
        txt_raw = meaning @ r.standard_normal((d, d))   # テキスト側の歪み

        # 学習対象: 画像・テキストそれぞれの射影（共有空間へ戻す役）
        Wi = r.standard_normal((d, d)) * 0.1
        Wt = r.standard_normal((d, d)) * 0.1

        def loss_of(Wi, Wt):
            return clip_loss(img_raw @ Wi, txt_raw @ Wt)[0]

        history = []
        eps = 1e-4
        for _ in range(steps):
            L = loss_of(Wi, Wt)
            history.append(L)
            # 有限差分で勾配を近似（教材用。実際は自動微分）
            gi = np.zeros_like(Wi)
            gt = np.zeros_like(Wt)
            for a in range(d):
                for b in range(d):
                    dWi = np.zeros_like(Wi); dWi[a, b] = eps
                    gi[a, b] = (loss_of(Wi + dWi, Wt) - L) / eps
                    dWt = np.zeros_like(Wt); dWt[a, b] = eps
                    gt[a, b] = (loss_of(Wi, Wt + dWt) - L) / eps
            Wi -= lr * gi
            Wt -= lr * gt

        final_logits = clip_loss(img_raw @ Wi, txt_raw @ Wt)[1]
        return history, final_logits

    hist, trained_logits = train_projection()
    (f"loss: {hist[0]:.3f} -> {hist[-1]:.3f}",)
    return hist, trained_logits


@app.cell
def _(hist, plt, trained_logits):
    def _plot_training():
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot(hist)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("CLIP loss")
        axes[0].set_title("contrastive loss goes down")
        im = axes[1].imshow(trained_logits, cmap="viridis")
        axes[1].set_title("learned similarity\n(diagonal lights up)")
        axes[1].set_xlabel("text"); axes[1].set_ylabel("image")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        return fig

    _plot_training()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        自前の射影だけで、無関係だった画像・テキスト特徴が **共有空間で結びつきました**。
        対角がくっきり光っています。これが CLIP が数億ペアの規模でやっていることの縮図です。

        ## 5. ゼロショット分類の考え方

        学習後の共有空間があれば、**未知のラベルでも分類** できます。手順は:

        1. 候補ラベルを文にする: `"a photo of a dog"`, `"a photo of a cat"`, ...
        2. それぞれテキストエンコーダでベクトル化
        3. 分類したい画像もベクトル化
        4. **一番コサイン類似度が高いラベル文** を答えにする

        「犬」を学習データのクラスとして持っていなくても、"a photo of a dog" という文さえ作れば分類できる。
        これがゼロショット。画像検索（元リポジトリ）は、この 3〜4 を「画像 vs 画像」でやっているだけです。
        """
    )
    return


@app.cell
def _(cosine_sim_matrix, np, rng, softmax):
    def zero_shot_demo():
        d = 16
        # 「意味」空間（学習済みの共有空間を模擬）
        concepts = {c: rng.standard_normal(d) for c in ["dog", "cat", "car"]}
        # ラベル文のベクトル（= その概念そのもの）
        label_names = list(concepts)
        text_vecs = np.stack([concepts[c] for c in label_names])
        # 分類したい画像 = "cat" に近いベクトル（少しノイズ）
        query_img = concepts["cat"] + 0.2 * rng.standard_normal(d)

        sims = cosine_sim_matrix(query_img[None, :], text_vecs)[0]
        probs = softmax(sims / 0.1)
        pred = label_names[int(np.argmax(sims))]
        return label_names, probs, pred

    zs_labels, zs_probs, zs_pred = zero_shot_demo()
    (f"predicted: {zs_pred}", dict(zip(zs_labels, zs_probs.round(3))))
    return zs_labels, zs_probs


@app.cell
def _(plt, zs_labels, zs_probs):
    def _plot_zs():
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.bar(zs_labels, zs_probs, color="tab:purple")
        ax.set_ylabel("probability")
        ax.set_title('zero-shot: which caption fits the image?')
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig

    _plot_zs()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## まとめ & 本物への橋渡し

        | CLIP の要素 | 意味 | 実際 |
        |------------|------|------|
        | デュアルエンコーダ | 画像用(ViT)・テキスト用を別々に | どちらも 02/03 の Transformer |
        | L2 正規化 + コサイン類似度 | 共有空間での「近さ」 | 検索・分類の物差し |
        | InfoNCE / clip_loss | 正しいペアを引き寄せ、他を離す | 温度 $\tau$ は学習可能 |
        | ゼロショット | ラベルを文にして比較 | 未知クラスに強い |

        **VLM への意味**: CLIP で訓練された **ViT は「言葉に結びついた視覚特徴」を出す**ようになります。
        だから多くの VLM（LLaVA, Qwen-VL …）は、視覚エンコーダに **CLIP/SigLIP 系の学習済み ViT** を使います。
        「すでに言語と対応づいた視覚特徴」を出発点にできるからです。

        ただし CLIP はまだ「画像とテキストの**近さ**を測る」だけで、**文章を生成** はできません。
        「画像を見て、それについて自由に喋る」には、視覚特徴を **LLM に流し込む** 必要があります。

        **次章 05（VLM / LLaVA）** で、ついに視覚エンコーダと大規模言語モデルを **projector** で接続し、
        「画像を見て言葉で答える」モデルを組み立てます。
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
