import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 06. Qwen-VL の中核技術 — そして本物へ

        ここまでで「画像を LLM に話させる」素朴な VLM（LLaVA, 05）ができました。
        最終章では **Qwen2-VL / Qwen2.5-VL / Qwen3-VL** が加えた 3 つの重要な改良を実装します。
        これらは 05 の弱点をちょうど埋めるものです。

        | 05 の弱点 | Qwen-VL の解 | 本章で実装 |
        |----------|-------------|-----------|
        | 位置情報がテキスト用の 1 次元のまま | **M-RoPE**（時間・縦・横を分けた回転位置埋め込み） | ✅ §1〜§2 |
        | 解像度が固定で細部が潰れる | **Naive Dynamic Resolution**（可変解像度） | ✅ §3 |
        | 高解像度でトークンが爆発 | **Vision Merger**（隣接パッチを統合） | ✅ §4 |

        最後に、これらを積んだ **本物の Qwen3-VL**（元リポジトリ `qwen-image-search` が使うモデル）へ橋渡しします。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(6)
    return np, plt, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. まず RoPE（回転位置埋め込み）

        02 では位置を sin/cos で **足し算** しました。現代の LLM（LLaMA, Qwen …）は代わりに
        **RoPE (Rotary Position Embedding)** を使います。アイデアは:

        > ベクトルを 2 次元ずつのペアに分け、**位置 $m$ に比例した角度だけ回転** させる

        次元ペア $i$ の回転角は $m\theta_i$、ただし $\theta_i = \text{base}^{-2i/d}$。
        低い次元は速く回り（細かい位置）、高い次元はゆっくり回る（大まかな位置）。

        RoPE の魔法は **相対位置** に効くこと。回転させた $q$（位置 $m$）と $k$（位置 $n$）の内積は、
        $m$ と $n$ の **差 $m-n$ だけ** に依存します。これを下で数値的に確認します。
        """
    )
    return


@app.cell
def _(np):
    def rope_angles(positions, dim, base=10000.0):
        """positions:(n,) → 各ペアの回転角 (n, dim/2)。"""
        i = np.arange(0, dim, 2)                 # 0,2,4,...
        inv_freq = base ** (-(i / dim))          # θ_i
        return positions[:, None] * inv_freq[None, :]  # (n, dim/2)

    def apply_rope(x, positions, base=10000.0):
        """x:(n,dim) を位置 positions で回転。"""
        n, dim = x.shape
        ang = rope_angles(positions, dim, base)  # (n, dim/2)
        cos, sin = np.cos(ang), np.sin(ang)
        x1, x2 = x[:, 0::2], x[:, 1::2]          # 偶数/奇数でペアを作る
        out = np.empty_like(x)
        out[:, 0::2] = x1 * cos - x2 * sin       # 回転行列を適用
        out[:, 1::2] = x1 * sin + x2 * cos
        return out

    return apply_rope, rope_angles


@app.cell
def _(apply_rope, np, plt, rng):
    def _demo_relative():
        dim = 32
        q = rng.standard_normal(dim)
        k = rng.standard_normal(dim)
        # k を固定位置 n=20 に置き、q の位置 m を動かす
        n_pos = 20
        k_rot = apply_rope(k[None, :], np.array([n_pos]))[0]
        ms = np.arange(0, 41)
        scores = []
        for m in ms:
            q_rot = apply_rope(q[None, :], np.array([m]))[0]
            scores.append(q_rot @ k_rot)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ms - n_pos, scores)
        ax.axvline(0, color="red", ls="--", alpha=0.6)
        ax.set_xlabel("relative position  (m - n)")
        ax.set_ylabel("q·k after RoPE")
        ax.set_title("RoPE makes the score depend on RELATIVE position\n(peak near offset 0)")
        fig.tight_layout()
        return fig

    _demo_relative()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        スコアが **相対位置 0 付近でピーク** になり、離れると減衰します。絶対位置ではなく「どれだけ離れているか」で
        効くのが RoPE。長い系列への外挿にも強く、これが LLM の標準になりました。

        ## 2. M-RoPE — 画像の 2 次元性を位置に取り戻す

        05 の問題: 画像をトークン列に並べると、本来 2 次元だった「どの行・どの列のパッチか」が
        **1 列の順番** に潰れてしまいます。左上のパッチと右下のパッチの空間関係が失われる。

        **Qwen2-VL の M-RoPE（Multimodal RoPE）** はこう解きます:

        > 位置 ID を **1 個のスカラー**ではなく **3 つ組 (t, h, w)** = (時間, 縦, 横) にする。
        > head 次元を 3 ブロックに分け、それぞれ t / h / w の位置で RoPE を回す。

        - **テキストトークン**: t = h = w = 系列位置（従来どおり 1 次元に退化）
        - **画像トークン**: t = フレーム番号, h = パッチの行, w = パッチの列（← 2 次元を保持！）

        まず各トークンに 3D 位置 ID を割り当てます。
        """
    )
    return


@app.cell
def _(np):
    def assign_mrope_positions(n_text_before, grid_hw, n_text_after):
        """
        テキスト → 画像(gh×gw) → テキスト の系列に (t,h,w) 位置IDを割り当てる。
        Qwen2-VL 方式を簡略化: 画像後のテキストは、画像が占めた最大IDの続きから再開する。
        戻り値: pos (N, 3) int, seg (N,) str
        """
        gh, gw = grid_hw
        pos, seg = [], []
        p = 0
        # 画像前テキスト: t=h=w=p
        for _ in range(n_text_before):
            pos.append([p, p, p]); seg.append("text"); p += 1
        # 画像: t は固定(=p), h=行, w=列。ここでは1フレーム画像
        t0 = p
        for r in range(gh):
            for c in range(gw):
                pos.append([t0, t0 + r, t0 + c]); seg.append("image")
        # 画像後テキストは、画像が使った最大IDの次から続ける
        p = t0 + max(gh, gw)
        for _ in range(n_text_after):
            pos.append([p, p, p]); seg.append("text"); p += 1
        return np.array(pos), seg

    mrope_pos, mrope_seg = assign_mrope_positions(
        n_text_before=3, grid_hw=(4, 4), n_text_after=4
    )
    ("sequence length:", len(mrope_seg), "/ position-id shape:", mrope_pos.shape)
    return mrope_pos, mrope_seg


@app.cell
def _(mrope_pos, mrope_seg, np, plt):
    def _plot_mrope_ids():
        fig, ax = plt.subplots(figsize=(9, 3.2))
        labels = ["t (time)", "h (row)", "w (col)"]
        colors = ["tab:green", "tab:red", "tab:blue"]
        x = np.arange(len(mrope_seg))
        for k in range(3):
            ax.plot(x, mrope_pos[:, k], "o-", color=colors[k], label=labels[k], ms=4)
        # 画像トークン範囲に帯
        img_idx = [i for i, s in enumerate(mrope_seg) if s == "image"]
        ax.axvspan(img_idx[0] - .5, img_idx[-1] + .5, color="orange", alpha=0.12)
        ax.set_xlabel("token index in sequence")
        ax.set_ylabel("position id")
        ax.set_title("M-RoPE position ids  (orange = image: h,w spread out in 2D; text: t=h=w)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    _plot_mrope_ids()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        グラフを見てください。**テキスト区間では t=h=w が重なって一直線**（1 次元）ですが、
        **画像区間では h（行）と w（列）が別々に広がり、2 次元の格子構造が位置 ID に刻まれています**。
        これで「隣のパッチ」「上のパッチ」という空間関係が RoPE を通じて attention に伝わります。

        次に、head 次元を (t, h, w) の 3 ブロックに分け、各ブロックを対応する位置で回す M-RoPE 本体を実装します。
        """
    )
    return


@app.cell
def _(apply_rope, np):
    def apply_mrope(x, pos3, base=10000.0):
        """
        x: (n, dim)  dim は 6 の倍数（3 セクション×各 even）を想定
        pos3: (n, 3) の (t,h,w) 位置ID
        head 次元を 3 分割し、t/h/w それぞれで RoPE を適用して連結する。
        """
        n, dim = x.shape
        sec = dim // 3
        # 各セクションが even になるよう調整
        sec -= sec % 2
        parts = []
        for k in range(3):
            xs = x[:, k * sec:(k + 1) * sec]
            parts.append(apply_rope(xs, pos3[:, k].astype(float), base))
        # 端数（割り切れない分）はそのまま
        rest = x[:, 3 * sec:]
        return np.concatenate(parts + [rest], axis=1)

    return (apply_mrope,)


@app.cell
def _(apply_mrope, mrope_pos, np, rng):
    # 動作確認: dim=36 (=3×12) のダミー特徴に M-RoPE を適用
    def _check_mrope():
        n = mrope_pos.shape[0]
        x = rng.standard_normal((n, 36))
        y = apply_mrope(x, mrope_pos)
        # 回転はノルムを保つ（各ペアが回転行列）→ セクション毎のノルムは不変
        return np.allclose(np.linalg.norm(x[:, :12], axis=1),
                           np.linalg.norm(y[:, :12], axis=1))

    ("M-RoPE preserves per-section norm (rotation):", _check_mrope())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        セクションごとのノルムが保たれています（回転だから当然）。ここが M-RoPE の実装の要です。

        > **Qwen3-VL への系譜**: Qwen2-VL でこの M-RoPE が導入され、Qwen2.5-VL で動画の時間軸（絶対時刻）に
        > 対応するよう t 成分が改良され、Qwen3-VL へと洗練されました。核は「位置を (t,h,w) に分ける」この発想です。

        ## 3. Naive Dynamic Resolution — 画像を潰さない

        05（と多くの旧来 VLM）は画像を固定サイズ（例 224×224）にリサイズしていました。
        Qwen-VL は **元の解像度に応じてパッチ数を変える**（naive dynamic resolution）。
        大きい画像はより多くのトークンになり、細部を保てます。トークン数はおおよそ画像の面積に比例します。
        """
    )
    return


@app.cell
def _(np, plt):
    def n_vision_tokens(h_px, w_px, patch=14, merge=2):
        """解像度 → 視覚トークン数。patch で割り、merge×merge で統合した後の数。"""
        gh, gw = h_px // patch, w_px // patch
        # merger で merge×merge をまとめる（§4）
        return (gh // merge) * (gw // merge)

    def _plot_dyn_res():
        sizes = [224, 336, 448, 672, 896, 1344]
        toks = [n_vision_tokens(s, s) for s in sizes]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(sizes, toks, "o-")
        for s, t in zip(sizes, toks):
            ax.annotate(str(t), (s, t), textcoords="offset points", xytext=(0, 6),
                        fontsize=8, ha="center")
        ax.set_xlabel("image side length (px, square)")
        ax.set_ylabel("# vision tokens (after 2x2 merge)")
        ax.set_title("Naive Dynamic Resolution:\ntokens grow with resolution (no fixed 224)")
        fig.tight_layout()
        return fig

    _plot_dyn_res()
    return (n_vision_tokens,)


@app.cell
def _(mo):
    mo.md(
        r"""
        固定解像度なら常に一定トークンですが、動的解像度では **大きい画像＝多いトークン＝多い情報**。
        小さいアイコンは数トークン、書類の高解像度スキャンは数千トークン、と入力に応じて伸縮します。

        ## 4. Vision Merger — トークン爆発を抑える

        高解像度だとパッチが増えすぎて LLM が重くなります。Qwen-VL は **隣接する 2×2 のパッチ特徴を
        1 つに統合**（concat して線形射影）し、視覚トークン数を **1/4** に減らします。
        情報は保ちつつ系列長を抑える賢い圧縮です。
        """
    )
    return


@app.cell
def _(np, rng):
    def patch_merger(feats_grid, W_merge):
        """
        feats_grid: (gh, gw, d) のパッチ特徴
        隣接 2x2 を concat(4d) → 線形射影(→d) して (gh/2, gw/2, d) に。
        """
        gh, gw, d = feats_grid.shape
        gh2, gw2 = gh // 2, gw // 2
        merged = np.zeros((gh2, gw2, d))
        for i in range(gh2):
            for j in range(gw2):
                block = feats_grid[2*i:2*i+2, 2*j:2*j+2, :].reshape(-1)  # (4d,)
                merged[i, j] = block @ W_merge                          # (d,)
        return merged

    def _demo_merge():
        gh, gw, d = 8, 8, 16
        feats = rng.standard_normal((gh, gw, d))
        W_merge = rng.standard_normal((4 * d, d)) / np.sqrt(4 * d)
        merged = patch_merger(feats, W_merge)
        before = gh * gw
        after = merged.shape[0] * merged.shape[1]
        return before, after, merged.shape

    _demo_merge()
    return (patch_merger,)


@app.cell
def _(mo):
    mo.md(
        r"""
        8×8 = 64 パッチが 4×4 = 16 トークンへ（1/4）。実際の Qwen-VL では、この merger の出力が
        §2 の M-RoPE 付き画像トークンとして LLM に入ります。

        ## 5. Qwen-VL のパイプライン全体像

        ここまでの部品を並べると、Qwen-VL の視覚→言語パイプラインになります:

        ```
          任意解像度の画像
              │  naive dynamic resolution (§3): 面積に応じてパッチ化
              ▼
          可変個のパッチ  ──[ ViT (03) with M-RoPE 2D位置 ]
              │  vision merger (§4): 2×2 を統合、トークン1/4
              ▼
          視覚トークン (可変個)
              │  projector (05) : LLM 空間へ
              ▼
          [ テキスト ⧺ 視覚トークン ] を M-RoPE(§2) 付きで
              │
              ▼
          LLM (02) が (t,h,w) 位置を意識して次語生成 / 埋め込み出力
        ```

        **01〜06 の全部品が一本に繋がりました。** attention（01）から始まり、
        Transformer（02）→ ViT（03）→ CLIP 的な視覚言語対応（04）→ VLM 統合（05）→
        Qwen-VL の改良（06）まで、一つの注意機構が形を変えて積み上がっています。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. 本物の Qwen3-VL へ — 元リポジトリとの接続

        この教材の出発点は `qwen-image-search`。その中身は **04 CLIP で学んだ画像検索そのもの** です:

        > 画像を Qwen3-VL-Embedding でベクトル化 → コサイン類似度で似た画像を探す

        つまり本教材は「なぜ画像がベクトルになり、なぜコサイン類似度で検索できるのか」を
        attention のレベルから再構築してきた、というわけです。実際のモデルを触るコード例（重いので任意実行）:

        ```python
        # pip install -e ".[qwen]"  してから（torch/transformers/qwen-vl-utils が入る）
        import torch
        from transformers import AutoModel, AutoProcessor

        model_id = "Qwen/Qwen3-VL-Embedding-2B"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).eval()

        # 画像を（可変解像度のまま）ベクトル化 —— §3 dynamic resolution が内部で効く
        from PIL import Image
        img = Image.open("photo.jpg")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = model(**inputs).pooler_output      # (1, d) の画像埋め込み
        emb = torch.nn.functional.normalize(emb, dim=-1)  # 04 の L2 正規化

        # 別画像との類似度 = 04 の cosine_sim_matrix と同じ
        # score = emb_query @ emb_db.T
        ```

        実際に触れる際のヒント（このモデルは §1〜§4 を全部積んでいます）:

        - `model.config` を見ると `vision_config`（ViT + merger）と `rope_scaling`（M-RoPE の section 設定）が確認できます
        - 高解像度画像を入れるほど視覚トークンが増える（§3）ことを、`inputs` のトークン数で観察できます
        - 出力埋め込みでの近傍探索が、元リポジトリ `app/` の検索 API の中身です

        下のセルは、環境に torch/transformers が入っているかだけを安全にチェックします（ダウンロードはしません）。
        """
    )
    return


@app.cell
def _():
    # 重い依存があるかの確認のみ（モデルDLはしない）
    def check_qwen_env():
        status = {}
        for pkg in ["torch", "transformers", "qwen_vl_utils"]:
            try:
                __import__(pkg)
                status[pkg] = "installed"
            except ImportError:
                status[pkg] = "not installed  (`pip install -e \".[qwen]\"` で入ります)"
        return status

    check_qwen_env()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## 全体のまとめ

        | 章 | 作ったもの | VLM の物語での役割 |
        |----|-----------|-------------------|
        | 01 | scaled dot-product / multi-head attention | すべての土台 |
        | 02 | Transformer ブロック, 位置エンコーディング | 言語モデルの背骨 |
        | 03 | ViT（patchify, CLS, patch embedding） | 画像を同じ機械に載せる |
        | 04 | CLIP（対照学習, ゼロショット, 検索） | 画像とテキストを同じ空間へ |
        | 05 | LLaVA（projector, マルチモーダル系列） | 画像を LLM に話させる |
        | 06 | RoPE / **M-RoPE** / 動的解像度 / **merger** | Qwen-VL の実用的改良 |

        **一貫した気づき**: 新しい構造の多くは「まったく新しい発明」ではなく、
        **attention という一つの部品を、新しい入力（画像）・新しい位置（2次元）・新しい規模（可変解像度）に
        適応させ続けた結果** です。だからこそ、01 で attention を手で書いた経験が、Qwen-VL まで一直線に効きます。

        ### 次に学ぶなら

        - **実際の重みで動かす**: §6 のコードで Qwen3-VL-Embedding をロードし、`app/` の検索を再現
        - **学習を入れる**: ここは全部ランダム重み。PyTorch + autograd で 04 の対照学習を本気で回すと理解が固まります
        - **動画・音声へ**: M-RoPE の t 成分が動画で活き、さらにマルチモーダルが広がります

        お疲れさまでした。attention から Qwen まで、VLM の一本道を歩き切りました 🎉
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
