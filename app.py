#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import itertools
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="多元回归自动分析工具", layout="wide")

st.title("📊 多元线性回归自动分析工具（含自动特征选择）")
st.write("上传 Excel 文件，每个 Sheet 将自动执行多元回归，并从所有特征组合中选出最优模型。")

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
                "model": model
            })

    # 选择 AIC 最小的模型为最佳
    best = sorted(best_models, key=lambda x: x["aic"])[0]
    return best

# ---------- 主程序 ----------
if uploaded:
    xls = pd.ExcelFile(uploaded)
    
    for sheet in xls.sheet_names:
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

        st.write("### 🔍 自动特征选择：正在遍历所有可能的特征组合…")

        best = best_subset_selection(X, y)
        model = best["model"]

        # ---------- 输出结果 ----------
        st.subheader("🏆 最佳模型（基于 AIC）")
        st.write(f"**最佳特征组合：** {list(best['features'])}")

        st.write("### 📈 回归结果")
        st.write(model.summary())

        # ---------- Pairplot ----------
        st.write("### 📊 散点矩阵图（Pairplot）")
        fig1 = sns.pairplot(df[list(best["features"]) + [y_col]])
        st.pyplot(fig1)

        # ---------- 残差图 ----------
        st.write("### 🟡 残差图")
        residuals = model.resid
        fitted = model.fittedvalues

        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=fitted, y=residuals, ax=ax2)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("预测值")
        ax2.set_ylabel("残差")
        st.pyplot(fig2)

        # ---------- QQ Plot ----------
        st.write("### 📐 QQ 图（检查误差正态性）")
        fig3 = sm.qqplot(residuals, line='45', fit=True)
        st.pyplot(fig3)

        # ---------- 预测值 vs 实际值 ----------
        st.write("### 🔵 实际值 vs 预测值")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=y, y=fitted, ax=ax4)
        ax4.set_xlabel("实际值")
        ax4.set_ylabel("预测值")
        st.pyplot(fig4)

    st.success("分析完成！请查看上方所有图表。")

