import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
subgroup_col = st.sidebar.selectbox("サブグループ列（None可）", options=["None"] + cols, index=1)
if subgroup_col == "None":
    subgroup_col = None
y_col = st.sidebar.selectbox("値列", options=cols, index=2)

# --- グラフ設定 ---
bar_width = st.sidebar.slider("Bar width", 0.5, 1.0, 0.7)
dot_size = st.sidebar.slider("Dot size", 2, 20, 6)
xtick_fontsize = st.sidebar.slider("X-axis font size", 8, 20, 12)
ytick_fontsize = st.sidebar.slider("Y-axis font size", 8, 20, 12)
fig_width = st.sidebar.slider("Figure width", 1, 10, 6)
fig_height = st.sidebar.slider("Figure height", 3, 12, 5)
y_max = st.sidebar.number_input("Y axis max", value=int(df[y_col].max() * 1.2))
y_step = st.sidebar.number_input("Y axis step", value=1)
line_width = st.sidebar.slider("Bar & SEM line width", 0.5, 5.0, 2.0)
sem_cap_length = st.sidebar.slider("SEM cap length", 2, 20, 6)
x_label = st.sidebar.text_input("X軸ラベル（空欄ならなし）", value=x_col)
y_label = st.sidebar.text_input("Y軸ラベル（空欄ならなし）", value=y_col)

# --- モード選択タブ ---
tab1, tab2 = st.tabs(["Bar + Swarm", "Swarmのみ"])
mode = tab1 if st.session_state.get("active_tab", "Bar + Swarm") == "Bar + Swarm" else tab2

# --- 色・凡例設定 ---
col1, col2 = st.columns([3, 1])

if subgroup_col is None:
    group_combos = [(g, None) for g in df[x_col].unique()]
else:
    group_combos = df[[x_col, subgroup_col]].drop_duplicates().values.tolist()

with col2:
    combo_labels = {}
    combo_colors = {}
    for i, (g, sg) in enumerate(group_combos):
        combo_key = f"{g} | {sg if sg is not None else ''}".strip()
        default_color = mcolors.to_hex(sns.color_palette("tab20c")[i % 20])
        combo_colors[(g, sg)] = st.color_picker(f"{combo_key} の色", value=default_color, key=f"color_{combo_key}")
        combo_labels[(g, sg)] = st.text_area(f"{combo_key} の凡例名（改行OK）", 
                                             value=f"{g}\n{sg}" if sg is not None else g, 
                                             key=f"label_{combo_key}")

# --- Group_Label列作成 ---
if subgroup_col is None:
    df["Group_Label"] = df[x_col].map({g: combo_labels.get((g, None), g) for g in df[x_col].unique()})
else:
    df["Group_Label"] = df.apply(
        lambda row: combo_labels.get((row[x_col], row[subgroup_col]), f"{row[x_col]}\n{row[subgroup_col]}"),
        axis=1
    )

# --- 並び順 ---
main_order = st.sidebar.multiselect("メイングループ順序", options=sorted(df[x_col].unique(), reverse=True),
                                    default=sorted(df[x_col].unique(), reverse=True))
if subgroup_col:
    sub_order = st.sidebar.multiselect("サブグループ順序", options=sorted(df[subgroup_col].unique()),
                                       default=sorted(df[subgroup_col].unique()))
else:
    sub_order = []

# --- グラフ描画関数 ---
def plot_graph(plot_mode):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if subgroup_col:
        grouped = df.groupby([x_col, subgroup_col])[y_col].agg(['mean', 'sem']).reset_index()
        grouped["Group_Label"] = grouped.apply(
            lambda row: combo_labels.get((row[x_col], row[subgroup_col]), f"{row[x_col]}\n{row[subgroup_col]}"),
            axis=1
        )
        grouped = grouped.sort_values([x_col, subgroup_col])
    else:
        grouped = df.groupby(x_col)[y_col].agg(['mean', 'sem']).reset_index()
        grouped["Group_Label"] = grouped[x_col].map({g: combo_labels.get((g, None), g) for g in df[x_col].unique()})
        grouped = grouped.sort_values(x_col)

    if plot_mode == "Bar + Swarm":
        # バー＋SEM
        for _, row in grouped.iterrows():
            edge_col = combo_colors.get((row[x_col], row[subgroup_col] if subgroup_col else None), "black")
            ax.bar(row["Group_Label"], row["mean"], width=bar_width, edgecolor=edge_col,
                   fill=False, linewidth=line_width)
            ax.errorbar(row["Group_Label"], row["mean"], yerr=row["sem"],
                        color='black', fmt='none',
                        capsize=sem_cap_length, capthick=line_width, elinewidth=line_width)

    elif plot_mode == "Swarmのみ":
        # 平均線のみ黒
        for _, row in grouped.iterrows():
            x_pos = main_order.index(row[x_col])
            ax.hlines(row["mean"], x_pos - 0.3, x_pos + 0.3, colors='black', linewidth=line_width)

    # スウォームプロット
    palette_map = {row["Group_Label"]: combo_colors.get((row[x_col], row[subgroup_col] if subgroup_col else None), "black")
                   for _, row in df.iterrows()}
    sns.swarmplot(
        data=df, x="Group_Label", y=y_col,
        hue="Group_Label",
        palette=palette_map, dodge=False,
        ax=ax, size=dot_size, legend=False
    )

    plt.setp(ax.get_xticklabels(), fontsize=xtick_fontsize)
    ax.set_xlabel(x_label if x_label else "", fontsize=xtick_fontsize)
    ax.set_ylabel(y_label if y_label else "", fontsize=ytick_fontsize)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + y_step, y_step))
    ax.tick_params(axis='y', labelsize=ytick_fontsize)
    st.pyplot(fig)

    # 保存
    save_format = st.radio("保存形式", options=["png", "pdf"])
    save_name = st.text_input("保存ファイル名（拡張子不要）", value="graph")
    if st.button("保存"):
        filename = f"{save_name}.{save_format}"
        fig.savefig(filename, bbox_inches='tight')
        st.success(f"{filename} として保存しました。")

# --- 描画 ---
active_tab = st.selectbox("モード選択", ["Bar + Swarm", "Swarmのみ"])
plot_graph(active_tab)

