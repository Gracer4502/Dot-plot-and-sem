import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colors as mcolors

# --- フォント設定 ---
plt.rcParams['font.size'] = 20

st.title("Dot Plot")

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

    # --- 色設定 ---
    unique_groups = df[x_col].unique()
    color_dict = {}
    for group in unique_groups:
        if sub_col:
            subs = df[df[x_col]==group][sub_col].unique()
            for idx, sub in enumerate(subs):
                default_color = sns.color_palette("tab10")[idx % 10]
                color_dict[(group, sub)] = st.color_picker(
                    f"{group}-{sub} の色",
                    value=mcolors.to_hex(default_color),
                    key=f"{group}-{sub}"
                )
        else:
            idx = list(unique_groups).index(group)
            default_color = sns.color_palette("tab10")[idx % 10]
            color_dict[(group, None)] = st.color_picker(
                f"{group} の色",
                value=mcolors.to_hex(default_color),
                key=f"{group}-main"
            )

    # --- 凡例名編集 ---
    legend_dict = {}
    for group in unique_groups:
        if sub_col:
            subs = df[df[x_col]==group][sub_col].unique()
            for sub in subs:
                legend_name = st.text_input(f"{group}-{sub} の凡例名", value=f"{group}-{sub}", key=f"legend-{group}-{sub}")
                legend_dict[(group, sub)] = legend_name
        else:
            legend_name = st.text_input(f"{group} の凡例名", value=f"{group}", key=f"legend-{group}")
            legend_dict[(group, None)] = legend_name

    # --- 重なり回避関数 ---
    def spread_y_vals(y_vals, x_center, spacing=0.05):
        """同じ y 値が重なる場合、左右にずらす"""
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
        # 元の順序に戻す
        x_positions[sorted_idx] += offsets
        return x_positions

    # --- 全体で均等配置するためのプロット群リスト ---
    plot_groups = []
    for group in unique_groups:
        if sub_col:
            subs = df[df[x_col]==group][sub_col].unique()
            for sub in subs:
                plot_groups.append((group, sub))
        else:
            plot_groups.append((group, None))

    # --- x 座標を均等割り当て ---
    n_total = len(plot_groups)
    x_coords = np.linspace(0, n_total-1, n_total)
    x_positions_dict = {plot_groups[i]: x_coords[i] for i in range(n_total)}

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(width, height))
    for key in plot_groups:
        group, sub = key
        group_df = df[df[x_col]==group] if sub is None else df[(df[x_col]==group) & (df[sub_col]==sub)]
        y_vals = pd.to_numeric(group_df[y_col], errors='coerce')
        x_center = x_positions_dict[key]
        x_vals = spread_y_vals(y_vals.values, x_center, spacing=0.05)
        ax.scatter(x_vals, y_vals, color=color_dict[key], alpha=1.0, s=scatter_size)
        # 平均とSEMを横線で表示
        y_mean = y_vals.mean()
        y_sem = y_vals.sem()
        if not np.isnan(y_mean):
            cap_width = 0.05  # キャップと同じ幅
            # SEM の縦線
            ax.vlines(x=x_center, ymin=y_mean - y_sem, ymax=y_mean + y_sem, color='black', linewidth=3)
            # SEM の横キャップ
            ax.hlines(y=y_mean - y_sem, xmin=x_center-cap_width, xmax=x_center+cap_width, color='black', linewidth=3)
            ax.hlines(y=y_mean + y_sem, xmin=x_center-cap_width, xmax=x_center+cap_width, color='black', linewidth=3)
    
    # --- X軸ラベル ---
    ax.set_xticks([x_positions_dict[k] for k in plot_groups])
    ax.set_xticklabels([legend_dict[k] for k in plot_groups], rotation=45, ha='right')

    ax.set_ylim(0, y_max)
    ax.set_ylabel(y_col)
    ax.set_xlabel(x_col)
    ax.set_yticks(np.arange(0, y_max + y_step, y_step))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig)
