import streamlit as st
import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import tempfile

# ===================== PDF 生成函数 =====================

def create_pdf_report(sheet_name, df_info_text, model_summary_text, image_bufs):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 标题
    story.append(Paragraph(f"<b>{sheet_name}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # 数据概览
    story.append(Paragraph("<b>📌 数据集信息</b>", styles["Heading2"]))
    story.append(Paragraph(df_info_text.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 12))

    # 模型概览
    story.append(Paragraph("<b>📌 最佳回归模型摘要</b>", styles["Heading2"]))
    story.append(Paragraph(model_summary_text.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(PageBreak())

    # 图像
    for title, img_buf in image_bufs:
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        story.append(Spacer(1, 8))

        # --- 修复关键点：将 BytesIO 写入临时文件 ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_buf.getvalue())
            tmp_path = tmp.name

        # 插图
        img = RLImage(tmp_path, width=15*cm)
        story.append(img)
        story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer


# ===================== 模型选择 =====================

def best_regression_model(df, target_col):
    X_cols = [c for c in df.columns if c != target_col]
    best_adjr2 = -999
    best_model = None
    best_features = None

    # 暴力搜索所有特征组合
    for r in range(1, len(X_cols)+1):
        for subset in itertools.combinations(X_cols, r):
            X = df[list(subset)]
            X = sm.add_constant(X)
            y = df[target_col]

            model = sm.OLS(y, X).fit()
            if model.rsquared_adj > best_adjr2:
                best_adjr2 = model.rsquared_adj
                best_model = model
                best_features = subset

    return best_model, best_features


# ===================== 图形绘制 =====================

def plot_regression(df, model, target_col, features):
    image_bufs = []

    # 预测值 + 残差
    df["pred"] = model.predict()
    df["resid"] = df[target_col] - df["pred"]

    # 散点图（多元模型 -> y vs predicted）
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=df["pred"], y=df[target_col], ax=ax)
    sns.lineplot(x=df["pred"], y=df["pred"], color="red", ax=ax)
    ax.set_title("Scatter Plot: y vs Predicted")
    buf1 = BytesIO()
    fig.savefig(buf1, format="png")
    plt.close(fig)

    image_bufs.append(("散点图 + 回归线", buf1))

    # 残差图
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=df["pred"], y=df["resid"], ax=ax)
    ax.axhline(0, color="red")
    ax.set_title("Residual Plot")
    buf2 = BytesIO()
    fig.savefig(buf2, format="png")
    plt.close(fig)

    image_bufs.append(("残差图", buf2))

    return image_bufs


# ===================== Streamlit 主程序 =====================

st.title("多元线性回归自动分析工具（含自动特征选择）")

uploaded_file = st.file_uploader("上传 Excel 文件（.xlsx）", type="xlsx")

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    for sheet_name in xls.sheet_names:
        st.header(f"📄 Sheet：{sheet_name}")

        df = xls.parse(sheet_name)
        st.dataframe(df)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("此 Sheet 数值列不足，无法回归。")
            continue

        target_col = numeric_cols[-1]  # 默认最后一列
        model, features = best_regression_model(df[numeric_cols], target_col)

        st.success(f"最佳特征组合：{features}")

        st.text(model.summary())

        # 绘图
        image_bufs = plot_regression(df[numeric_cols].copy(), model, target_col, features)

        # PDF 导出
        df_info_text = str(df.describe())
        model_summary_text = str(model.summary())

        pdf_buffer = create_pdf_report(sheet_name, df_info_text, model_summary_text, image_bufs)

        st.download_button(
            label=f"📥 下载报告（{sheet_name}.pdf）",
            data=pdf_buffer,
            file_name=f"{sheet_name}_report.pdf",
            mime="application/pdf"
        )



