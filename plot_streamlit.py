import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")

# =========================================================================
#                           データアップロード
# =========================================================================
uploaded_file = st.file_uploader("CSVまたはExcelをアップロード", type=['csv', 'xlsx'])
if uploaded_file is None:
    st.warning("ファイルをアップロードしてください")
    st.stop()
else:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# =========================================================================
#                             データフィルタリング
# =========================================================================
st.sidebar.markdown("### データフィルター設定")

filter_cols = st.sidebar.multiselect(
    "フィルタする列を選択（複数可）", 
    options=df.columns.tolist()
)

df_filtered = df.copy()

for c in filter_cols:
    unique_vals = sorted(df[c].dropna().unique().tolist())
    selected_vals = st.sidebar.multiselect(
        f"{c} の値を含める", 
        options=unique_vals, 
        default=unique_vals
    )
    df_filtered = df_filtered[df_filtered[c].isin(selected_vals)]

if df_filtered.empty:
    st.error("フィルター条件に一致するデータがありません。条件を変更してください。")
    st.stop()

# =========================================================================
#                             カラム選択
# =========================================================================
cols = df_filtered.columns.tolist()
x_col = st.sidebar.selectbox("メイングループ列（例: Genotype）", options=cols, index=0)

subgroup_col = st.sidebar.selectbox("サブグループ列（None可）", options=["None"] + cols)
if subgroup_col == "None":
    subgroup_col = None

y_col = st.sidebar.selectbox("値列", options=cols, index=1)

# =========================================================================
#                             グラフ設定
# =========================================================================
bar_width = st.sidebar.slider("Bar width", 0.1, 1.0, 0.7)
dot_size = st.sidebar.slider("Dot size", 2, 20, 6)
xtick_fontsize = st.sidebar.slider("X-axis font size", 8, 20, 12)
ytick_fontsize = st.sidebar.slider("Y-axis font size", 8, 20, 12)
fig_width = st.sidebar.number_input("Figure width", value=6.0, min_value=0.1, step=0.1, format="%.2f")
fig_height = st.sidebar.number_input("Figure height", value=6.0, min_value=0.1, step=0.1, format="%.2f")

y_max = st.sidebar.number_input("Y axis max", value=int(df_filtered[y_col].max() * 1.2))
y_step = st.sidebar.number_input("Y axis step", value=1.0, min_value=0.0001, step=0.1, format="%.4f")

line_width = st.sidebar.slider("Bar & SEM line width", 0.5, 10.0, 3.0)
sem_cap_length = st.sidebar.slider("SEM cap length", 2, 20, 6)

x_label = st.sidebar.text_input("X軸ラベル", value=x_col)
y_label = st.sidebar.text_input("Y軸ラベル", value=y_col)

# =========================================================================
#                 グループコンボ（メイン × サブ）を取得
# =========================================================================
if subgroup_col is None:
    group_combos = [(g, None) for g in df_filtered[x_col].unique()]
else:
    group_combos = df_filtered[[x_col, subgroup_col]].drop_duplicates().values.tolist()

# =========================================================================
#                      色と表示名（凡例名）の設定
# =========================================================================
col1, col2 = st.columns([3, 1])

with col2:
    combo_labels = {}
    combo_colors = {}

    for i, (g, sg) in enumerate(group_combos):
        combo_key = f"{g} | {sg if sg is not None else ''}".strip()
        default_color = mcolors.to_hex(sns.color_palette("tab20c")[i % 20])
        combo_colors[(g, sg)] = st.color_picker(
            f"{combo_key} の色", 
            value=default_color, 
            key=f"color_{combo_key}"
        )
        combo_labels[(g, sg)] = st.text_area(
            f"{combo_key} の凡例名（改行OK）",
            value=f"{g}\n{sg}" if sg is not None else g,
            key=f"label_{combo_key}"
        )

# =========================================================================
#                             Group_Label 作成
# =========================================================================
if subgroup_col is None:
    df_filtered["Group_Label"] = df_filtered[x_col].map(
        {g: combo_labels.get((g, None), g) for g in df_filtered[x_col].unique()}
    )
else:
    df_filtered["Group_Label"] = df_filtered.apply(
        lambda row: combo_labels.get((row[x_col], row[subgroup_col]),
                                     f"{row[x_col]}\n{row[subgroup_col]}"),
        axis=1
    )

# =========================================================================
#                         グループ順序（任意設定）
# =========================================================================
main_order_default = sorted(df_filtered[x_col].unique(), reverse=True)
main_order = st.sidebar.multiselect(
    "メイングループ順序",
    options=main_order_default,
    default=main_order_default
)

if subgroup_col:
    sub_order_default = sorted(df_filtered[subgroup_col].unique())
    sub_order = st.sidebar.multiselect(
        "サブグループ順序",
        options=sub_order_default,
        default=sub_order_default
    )
else:
    sub_order = []

# =========================================================================
#                             グラフ描画
# =========================================================================
with col1:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # --- 平均・SEM ---
    if subgroup_col:
        grouped = df_filtered.groupby([x_col, subgroup_col])[y_col].agg(['mean', 'sem']).reset_index()
        grouped["Group_Label"] = grouped.apply(
            lambda row: combo_labels.get((row[x_col], row[subgroup_col]),
                                         f"{row[x_col]}\n{row[subgroup_col]}"),
            axis=1
        )
    else:
        grouped = df_filtered.groupby([x_col])[y_col].agg(['mean', 'sem']).reset_index()
        grouped["Group_Label"] = grouped[x_col].map(
            {g: combo_labels.get((g, None), g) for g in df_filtered[x_col].unique()}
        )

    # --- 棒 + SEM ---
    for _, row in grouped.iterrows():
        edge_col = combo_colors.get((row[x_col], row[subgroup_col] if subgroup_col else None), "black")

        ax.bar(
            row["Group_Label"], row["mean"],
            width=bar_width, edgecolor=edge_col,
            fill=False, linewidth=line_width
        )

        ax.errorbar(
            row["Group_Label"], row["mean"],
            yerr=row["sem"], color='black', fmt='none',
            capsize=sem_cap_length, capthick=line_width, elinewidth=line_width
        )

    # --- スウォーム ---
    palette_map = {
        row["Group_Label"]: combo_colors.get(
            (row[x_col], row[subgroup_col] if subgroup_col else None), "black"
        )
        for _, row in df_filtered.iterrows()
    }

    sns.swarmplot(
        data=df_filtered, x="Group_Label", y=y_col,
        hue="Group_Label", palette=palette_map,
        dodge=False, ax=ax, size=dot_size,
        legend=False
    )

    # --- 軸設定 ---
    plt.setp(ax.get_xticklabels(), fontsize=xtick_fontsize)
    ax.set_xlabel(x_label if x_label else "", fontsize=xtick_fontsize)
    ax.set_ylabel(y_label if y_label else "", fontsize=ytick_fontsize)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + y_step, y_step))
    ax.tick_params(axis='y', labelsize=ytick_fontsize)

    st.pyplot(fig)

    # --- 保存 ---
    save_format = st.radio("保存形式", options=["png", "pdf"])
    save_name = st.text_input("保存ファイル名（拡張子不要）", value="graph")
    if st.button("保存"):
        filename = f"{save_name}.{save_format}"
        fig.savefig(filename, bbox_inches='tight')
        st.success(f"{filename} として保存しました。")
