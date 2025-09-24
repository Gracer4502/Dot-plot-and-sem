import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io

st.set_page_config(layout="wide")

# --- データアップロード ---
uploaded_file = st.file_uploader("CSVまたはExcelをアップロード", type=['csv', 'xlsx'])
if uploaded_file is None:
    st.warning("ファイルをアップロードしてください")
    st.stop()
else:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# --- カラム選択 ---
cols = df.columns.tolist()
x_col = st.sidebar.selectbox("メイングループ列", options=cols, index=0)
subgroup_col = st.sidebar.selectbox("サブグループ列", options=cols, index=1)
y_col = st.sidebar.selectbox("値列", options=cols, index=2)

# --- グラフ設定 ---
bar_width = st.sidebar.slider("Bar width", 0.5, 1.0, 0.7)
dot_size = st.sidebar.slider("Dot size", 2, 20, 6)
xtick_fontsize = st.sidebar.slider("X-axis font size", 8, 20, 12)
ytick_fontsize = st.sidebar.slider("Y-axis font size", 8, 20, 12)
fig_width = st.sidebar.slider("Figure width", 1, 10, 5)
fig_height = st.sidebar.slider("Figure height", 3, 12, 5)

# Y軸 maxとステップ
y_max = st.sidebar.number_input("Y axis max", value=int(df[y_col].max() * 1.2))
y_step = st.sidebar.number_input("Y axis step", value=1)

# 棒の枠線太さとSEMの太さ
line_width = st.sidebar.slider("Bar & SEM line width", 0.5, 5.0, 2.0)
sem_cap_size = line_width
sem_cap_length = st.sidebar.slider("SEM cap length", 2, 20, 6)

# 軸ラベル
x_label = st.sidebar.text_input("X軸ラベル（空欄ならなし）", value=x_col)
y_label = st.sidebar.text_input("Y軸ラベル（空欄ならなし）", value=y_col)

# --- メイングループ×サブグループ色・ラベル設定 ---
col1, col2 = st.columns([3, 1])
with col2:
    group_combos = df[[x_col, subgroup_col]].drop_duplicates().values.tolist()
    combo_labels = {}
    combo_colors = {}

    for i, (g, sg) in enumerate(group_combos):
        combo_key = f"{g} | {sg}"
        # サブグループを薄い順に色設定
        default_color = mcolors.to_hex(sns.color_palette("tab20")[i % 20])
        combo_colors[(g, sg)] = st.color_picker(f"{combo_key} の色", value=default_color, key=f"color_{combo_key}")
        combo_labels[(g, sg)] = st.text_area(f"{combo_key} の凡例名（改行OK）", value=f"{g}\n{sg}", key=f"label_{combo_key}")

# --- Group_Label列 ---
df["Group_Label"] = df.apply(
    lambda row: combo_labels.get((row[x_col], row[subgroup_col]), f"{row[x_col]}\n{row[subgroup_col]}"),
    axis=1
)

# --- デフォルト順序 ---
main_order_default = sorted(df[x_col].unique(), reverse=True)
sub_order_default = sorted(df[subgroup_col].unique())
main_order = st.sidebar.multiselect("メイングループ順序", options=main_order_default, default=main_order_default)
sub_order = st.sidebar.multiselect("サブグループ順序", options=sub_order_default, default=sub_order_default)

# --- グラフ描画 ---
with col1:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 平均+SEM
    grouped = df.groupby([x_col, subgroup_col])[y_col].agg(['mean', 'sem']).reset_index()
    grouped["Group_Label"] = grouped.apply(
        lambda row: combo_labels.get((row[x_col], row[subgroup_col]), f"{row[x_col]}\n{row[subgroup_col]}"),
        axis=1
    )
    grouped[x_col] = pd.Categorical(grouped[x_col], categories=main_order, ordered=True)
    grouped[subgroup_col] = pd.Categorical(grouped[subgroup_col], categories=sub_order, ordered=True)
    grouped = grouped.sort_values([x_col, subgroup_col])

    for _, row in grouped.iterrows():
        edge_col = combo_colors.get((row[x_col], row[subgroup_col]), "black")
        ax.bar(row["Group_Label"], row["mean"], width=bar_width, edgecolor=edge_col,
               fill=False, linewidth=line_width)
        ax.errorbar(row["Group_Label"], row["mean"], yerr=row["sem"],
                    color='black', capsize=sem_cap_length, fmt='none',
                    linewidth=line_width, elinewidth=line_width)  # キャップ太さに反映

    # スウォームプロット
    palette_map = {row["Group_Label"]: combo_colors.get((row[x_col], row[subgroup_col]), "black")
                   for _, row in df.iterrows()}

    sns.swarmplot(
        data=df, x="Group_Label", y=y_col,
        hue="Group_Label",
        palette=palette_map, dodge=False,
        ax=ax, size=dot_size, legend=False
    )

    # 軸設定
    plt.setp(ax.get_xticklabels(), fontsize=xtick_fontsize)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + y_step, y_step))
    ax.tick_params(axis='y', labelsize=ytick_fontsize)

    # 軸ラベル（フォントサイズ連動）
    ax.set_xlabel(x_label if x_label else "", fontsize=xtick_fontsize)
    ax.set_ylabel(y_label if y_label else "", fontsize=ytick_fontsize)

    st.pyplot(fig)

    # --- 保存用 ---
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', bbox_inches='tight')
    buf_png.seek(0)
    st.download_button("Download PNG", data=buf_png, file_name="dotplot.png", mime="image/png")

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
    buf_pdf.seek(0)
    st.download_button("Download PDF", data=buf_pdf, file_name="dotplot.pdf", mime="application/pdf")

