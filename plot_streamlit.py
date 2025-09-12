import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colors as mcolors
import io

# --- フォント設定 ---
plt.rcParams['font.size'] = 20

st.title("Excel Dot Plot")

# --- Excelアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("データプレビュー", df)

    # --- 列選択 ---
    all_columns = df.columns.tolist()
    x_col = st.selectbox("メイングループ列", all_columns)
    y_col = st.selectbox("値の列", all_columns)
    sub_col_option = st.selectbox("サブグループ列（任意）", [None] + all_columns)
    sub_col = sub_col_option if sub_col_option != "None" else None

    # --- ドットサイズ ---
    scatter_size = st.slider("ドットサイズ", 10, 500, 100, 10)

    # --- 縦横比 ---
    width = st.slider("グラフ幅", 4, 20, 10)
    height = st.slider("グラフ高さ", 4, 20, 12)

    # --- Y軸設定 ---
    y_numeric = pd.to_numeric(df[y_col], errors='coerce').dropna()
    if len(y_numeric) == 0:
        st.error(f"{y_col} に数値データがありません。")
        st.stop()
    y_min_val, y_max_val = y_numeric.min(), y_numeric.max()
    y_max = st.number_input("縦軸最大値", value=float(y_max_val), step=1.0)
    y_step = st.number_input("縦軸目盛り間隔", value=(y_max - y_min_val)/10, step=0.1)

    # --- プロット群作成 ---
    plot_groups = []
    for g in df[x_col].unique():
        if sub_col:
            for s in df[df[x_col]==g][sub_col].unique():
                plot_groups.append((g, s))
        else:
            plot_groups.append((g, None))

    # --- session_stateで色と凡例名保持 ---
    for key in plot_groups:
        if key not in st.session_state:
            default_color = sns.color_palette("tab10")[0] if key[0] == df[x_col].unique()[0] else sns.color_palette("tab10")[1] if key[0] == df[x_col].unique()[1] else sns.color_palette("hsv", 64)[0]
            st.session_state[key] = {"color": mcolors.to_hex(default_color),
                                     "legend": f"{key[0]}-{key[1]}" if key[1] else f"{key[0]}"}

    # --- 重なり回避関数 ---
    def spread_y_vals(y_vals, x_center, spacing=0.05):
        x_positions = np.full(len(y_vals), x_center)
        sorted_idx = np.argsort(y_vals)
        offsets = np.zeros(len(y_vals))
        y_sorted = y_vals[sorted_idx]
        for i in range(1, len(y_sorted)):
            same_y_idx = np.where(y_sorted[:i] == y_sorted[i])[0]
            if len(same_y_idx) > 0:
                offsets[i] = ((len(same_y_idx)+1)//2) * spacing
                if len(same_y_idx) % 2 == 0:
                    offsets[i] *= -1
        x_positions[sorted_idx] += offsets
        return x_positions

    # --- 横並びレイアウト ---
    graph_col, control_col = st.columns([3, 1])

    # --- x 座標を均等割り当て ---
    n_total = len(plot_groups)
    x_coords = np.linspace(0, n_total-1, n_total)
    x_positions_dict = {plot_groups[i]: x_coords[i] for i in range(n_total)}

    # --- グラフ描画 ---
    with graph_col:
        fig, ax = plt.subplots(figsize=(width, height))
        for key in plot_groups:
            group, sub = key
            group_df = df[df[x_col]==group] if sub is None else df[(df[x_col]==group) & (df[sub_col]==sub)]
            y_vals = pd.to_numeric(group_df[y_col], errors='coerce')
            x_center = x_positions_dict[key]
            x_vals = spread_y_vals(y_vals.values, x_center, spacing=0.05)
            ax.scatter(x_vals, y_vals, color=st.session_state[key]["color"], alpha=1.0, s=scatter_size)

            # 平均とSEMを描画
            y_mean = y_vals.mean()
            y_sem = y_vals.sem()
            if not np.isnan(y_mean):
                cap_width = 0.1  # 横線の長さ（左右に0.1ずつ広げる）
                ax.vlines(x_center, y_mean - y_sem, y_mean + y_sem, color='black', lw=3)  # 縦線
                ax.hlines([y_mean - y_sem, y_mean + y_sem], x_center - cap_width/2, x_center + cap_width/2, color='black', lw=3)  # キャップ

        ax.set_xticks([x_positions_dict[k] for k in plot_groups])
        ax.set_xticklabels([st.session_state[k]["legend"] for k in plot_groups], rotation=45, ha='right')
        ax.set_ylim(0, y_max)
        ax.set_ylabel(y_col)
        ax.set_xlabel(x_col)
        ax.set_yticks(np.arange(0, y_max + y_step, y_step))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # --- PNG / PDFダウンロード ---
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", bbox_inches="tight")
        buf_png.seek(0)
        buf_pdf = io.BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        buf_pdf.seek(0)

        st.download_button("PNGで保存", data=buf_png, file_name="dot_plot.png", mime="image/png")
        st.download_button("PDFで保存", data=buf_pdf, file_name="dot_plot.pdf", mime="application/pdf")

    # --- 色と凡例名編集パネル ---
    with control_col:
        st.write("### 色と凡例名の選択")
        for key in plot_groups:
            group, sub = key
            st.write(f"**{group}-{sub}**" if sub else f"**{group}**")
            selected_color = st.color_picker("色", value=st.session_state[key]["color"], key=f"color_{key}")
            st.session_state[key]["color"] = selected_color
            legend_name = st.text_input("凡例名", value=st.session_state[key]["legend"], key=f"legend_{key}")
            st.session_state[key]["legend"] = legend_name
