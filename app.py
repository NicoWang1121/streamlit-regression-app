# app.py
import streamlit as st
import pandas as pd
import itertools
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

st.set_page_config(page_title="多元回归自动分析工具（含PDF下载）", layout="wide")

st.title("📊 多元线性回归自动分析工具（含自动特征选择 & PDF 报告）")
st.write("上传 Excel 文件，系统会自动执行多元回归（Best Subset），并生成可下载的 PDF 报告。")

uploaded = st.file_uploader("上传 Excel 文件（.xlsx）", type=["xlsx"])

# ---------- 子函数：遍历所有特征组合 ----------
def best_subset_selection(X, y):
    best_models = []

    for k in range(1, len(X.columns) + 1):
        for combo in itertools.combinations(X.columns, k):
            X_subset = X[list(combo)]
            X_subset = sm.add_constant(X_subset)
            model = sm.OLS(y, X_subset).fit()

            best_models.append({
                "features": combo,
                "aic": model.aic,
                "bic": model.bic,
                "adj_r2": model.rsquared_adj,
                "r2": model.rsquared,
                "model": model
            })

    # 选择 AIC 最小的模型为最佳
    best = sorted(best_models, key=lambda x: x["aic"])[0]
    return best

# ---------- 子函数：把 matplotlib 图保存到 BytesIO PNG ----------
def fig_to_png_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------- 子函数：生成 PDF（使用 reportlab Platypus） ----------
def create_pdf_report(sheet_name, df_info_text, model_summary_text, image_bufs):
    """
    - sheet_name: str
    - df_info_text: str (数据说明)
    - model_summary_text: str (模型关键信息字符串)
    - image_bufs: list of tuples: (title, BytesIO_png)
    returns: BytesIO of PDF
    """
    pdf_buf = BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # 封面
    story.append(Paragraph(f"<b>回归分析报告 — {sheet_name}</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(df_info_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>模型概览</b>", styles['Heading2']))
    story.append(Paragraph(model_summary_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(PageBreak())

    # 每张图占一页（图上方写标题）
    for title, img_buf in image_bufs:
        story.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
        story.append(Spacer(1, 8))
        # reportlab Image 可以接受 BytesIO
        img = RLImage(img_buf, width=16*cm, preserveAspectRatio=True)
        story.append(img)
        story.append(PageBreak())

    # Build PDF
    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf

# ---------- 主程序 ----------
if uploaded:
    xls = pd.ExcelFile(uploaded)
    sheet_names = xls.sheet_names

    for sheet in sheet_names:
        st.header(f"📄 Sheet：{sheet}")

        df = pd.read_excel(uploaded, sheet_name=sheet).dropna()

        if df.shape[1] < 2:
            st.warning("列数不足（至少需要 2 列：特征 + 目标）。")
            continue

        # 最后一列为 Y，其余为 X
        y_col = df.columns[-1]
        X_cols = df.columns[:-1]

        X = df[X_cols]
        y = df[y_col]

        st.write("### 🔍 自动特征选择：正在遍历所有可能的特征组合…（可能对列数较多时较慢）")
        best = best_subset_selection(X, y)
        model = best["model"]

        # ---------- 输出结果（页面显示） ----------
        st.subheader("🏆 最佳模型（基于 AIC）")
        st.write(f"**最佳特征组合：** {list(best['features'])}")
        st.write("### 📈 回归结果（摘要）")
        st.text(model.summary().as_text())

        # ---------- 生成图：Pairplot、残差、QQ、实际vs预测 ----------
        st.write("### 🔎 可视化图（已生成，并会包含在 PDF 中）")

        image_bufs = []

        # 1) Pairplot（注意：pairplot 会自行新开 figure）
        try:
            pairplot_fig = sns.pairplot(df[list(best["features"]) + [y_col]])
            buf_pair = fig_to_png_bytes(pairplot_fig.fig)
            image_bufs.append(("散点矩阵图（Pairplot）", buf_pair))
            st.pyplot(pairplot_fig)
        except Exception as e:
            st.warning(f"Pairplot 生成失败：{e}")

        # 2) 散点图 + 回归线（Actual vs Predicted）
        fitted = model.fittedvalues
        fig_ap, ax_ap = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=y, y=fitted, ax=ax_ap)
        ax_ap.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax_ap.set_xlabel("实际值")
        ax_ap.set_ylabel("预测值")
        ax_ap.set_title("实际值 vs 预测值")
        buf_ap = fig_to_png_bytes(fig_ap)
        image_bufs.append(("实际值 vs 预测值", buf_ap))
        st.pyplot(fig_ap)

        # 3) 残差图
        residuals = model.resid
        fig_res, ax_res = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=fitted, y=residuals, ax=ax_res)
        ax_res.axhline(0, color="red", linestyle="--")
        ax_res.set_xlabel("预测值")
        ax_res.set_ylabel("残差")
        ax_res.set_title("残差图")
        buf_res = fig_to_png_bytes(fig_res)
        image_bufs.append(("残差图", buf_res))
        st.pyplot(fig_res)

        # 4) QQ 图
        fig_qq = sm.qqplot(residuals, line='45', fit=True)
        buf_qq = fig_to_png_bytes(fig_qq)
        image_bufs.append(("QQ 图（检验残差正态性）", buf_qq))
        st.pyplot(fig_qq)

        # ---------- 准备报告中的文字信息 ----------
        df_info_text = (
            f"Sheet 名称: {sheet}\n"
            f"样本数: {df.shape[0]}\n"
            f"特征数量: {len(X_cols)}\n"
            f"因变量: {y_col}\n"
        )

        # 模型主要统计量文本化（取 coef、pvalues 等）
        coef_table = model.params.to_frame(name='coef')
        coef_table['pvalue'] = model.pvalues
        coef_table['stderr'] = model.bse
        coef_lines = []
        for idx in coef_table.index:
            coef_lines.append(f"{idx}: coef={coef_table.loc[idx,'coef']:.4f}, stderr={coef_table.loc[idx,'stderr']:.4f}, p={coef_table.loc[idx,'pvalue']:.4g}")
        coef_text = "\n".join(coef_lines)

        model_summary_text = (
            f"最佳特征组合: {list(best['features'])}\n\n"
            f"R²: {model.rsquared:.4f}\n"
            f"Adjusted R²: {model.rsquared_adj:.4f}\n"
            f"AIC: {model.aic:.4f}\n"
            f"BIC: {model.bic:.4f}\n\n"
            f"系数与显著性:\n{coef_text}\n\n"
            "（详细的回归表请参见上方 Summary）"
        )

        # ---------- 生成 PDF（BytesIO） ----------
        pdf_buf = create_pdf_report(sheet, df_info_text, model_summary_text, image_bufs)

        # ---------- 在 Streamlit 页面提供下载 ----------
        st.download_button(
            label="⬇️ 下载本 Sheet 的 PDF 报告",
            data=pdf_buf.getvalue(),
            file_name=f"{sheet}_regression_report.pdf",
            mime="application/pdf"
        )

    st.success("全部 Sheet 的分析与 PDF 报告已生成。")


