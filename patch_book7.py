#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_book7.py
把《鸢尾花书》Book7 原版 notebook 修补为可在「现代 Python 3.12 + 离线数据」环境运行。
用法: 在仓库根目录执行  python patch_book7.py
已把所有已知修复编码为字符串替换（幂等：原串不存在则跳过并提示）。
"""
import json, os, glob, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
report = {"patched": 0, "skipped": 0}

def read_nb(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def write_nb(path, nb):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

def patch_cells(nb, subs):
    """对每个 code cell 依次做字符串替换; subs=[(old,new),...]"""
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        new = src
        for old, rep in subs:
            if old in new:
                new = new.replace(old, rep)
        if new != src:
            c["source"] = new.splitlines(True)
            n += 1
    return n

def patch_file(path, subs, skip_on=None):
    """修补单个 notebook 文件"""
    if not os.path.exists(path):
        report["skipped"] += 1
        return
    if skip_on and skip_on in open(path, encoding="utf-8").read():
        return  # 已补过
    nb = read_nb(path)
    k = patch_cells(nb, subs)
    write_nb(path, nb)
    report["patched"] += k
    print(f"  [ok] {os.path.basename(path)} (改 {k} cell)")

# ============================================================
# 0) 若 Ch02 附带数据为空，则生成合成数据(几何布朗运动)
# ============================================================
raw02 = os.path.join(ROOT, "Book7_Ch02_Python_Codes", "y_x_df_raw.pkl")
try:
    d = pd.read_pickle(raw02)
    empty = (d.shape[0] == 0)
except Exception:
    empty = True
if empty:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    n = len(idx)
    def gbm(s0, mu, sig):
        p = s0 * np.exp(np.cumsum(rng.normal(mu, sig, n)))
        return p
    prices = {"AAPL": gbm(300, 0.0004, 0.02), "^GSPC": gbm(3000, 0.0004, 0.012)}
    tickers = ["AAPL", "^GSPC"]
    mi = pd.MultiIndex.from_product([["Adj Close","Close","High","Low","Open","Volume"], tickers])
    df = pd.DataFrame(index=idx, columns=mi)
    for t in tickers:
        p = prices[t]
        df[("Adj Close", t)] = p; df[("Close", t)] = p
        df[("High", t)] = p*1.01; df[("Low", t)] = p*0.99
        df[("Open", t)] = p*0.995; df[("Volume", t)] = rng.integers(5e6, 1.5e7, n)
    df.to_pickle(raw02)
    print("  [数据] Ch02 生成合成 y_x_df_raw.pkl", df.shape)
else:
    print("  [数据] Ch02 y_x_df_raw.pkl 已有数据，跳过")

# ============================================================
# Ch02: 回归入门（联网下载 -> 读本地 pkl）
# ============================================================
dl2 = "y_x_df = yf.download(['AAPL','^GSPC'], start='2020-01-01', end='2020-12-31')"
print("[Ch02] 回归入门")
patch_file("Book7_Ch02_Python_Codes/Bk7_Ch02_01.ipynb",
           [("y_x_df_raw = yf.download(['AAPL','^GSPC'], start='2020-01-01', end='2020-12-31')",
             "y_x_df_raw = pd.read_pickle('y_x_df_raw.pkl')")])
for f in ["Bk7_Ch02_02.ipynb", "Bk7_Ch02_03.ipynb"]:
    patch_file(f"Book7_Ch02_Python_Codes/{f}",
               [(dl2, "y_x_df = pd.read_pickle('y_x_df_raw.pkl')")])

# ============================================================
# Ch03: 多元线性回归
# ============================================================
print("[Ch03] 多元线性回归")
patch_file("Book7_Ch03_Python_Codes/Bk7_Ch03_01.ipynb",
           [("""y_X_df = yf.download(['AAPL','MCD','^GSPC'], 
                     start='2020-01-01', 
                     end='2020-12-31')

y_X_df.to_pickle('y_X_df.pkl')""",
             "y_X_df = pd.read_pickle('y_X_df.pkl')")])
for f in ["Bk7_Ch03_02.ipynb", "Bk7_Ch03_03.ipynb"]:
    patch_file(f"Book7_Ch03_Python_Codes/{f}",
               [("stock_levels_df = yf.download(tickers, start='2020-07-01', end='2020-12-31')",
                 "stock_levels_df = pd.read_pickle('stock_levels_df.pkl')")])
# Ch03_02 原书bug: 两处 print 顺序写反
f = "Book7_Ch03_Python_Codes/Bk7_Ch03_02.ipynb"
nb = read_nb(f)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "VIF(X_df.values" in src and "print(VIF_X_no_1_df)" in src:
            c["source"] = src.replace("print(VIF_X_no_1_df)", "print(VIF_X_df)").splitlines(True)
        elif "VIF(X_df_no_1.values" in src and "print(VIF_X_df)" in src:
            c["source"] = src.replace("print(VIF_X_df)", "print(VIF_X_no_1_df)").splitlines(True)
write_nb(f, nb)
print("  [ok] Ch03_02 print顺序修正")

# ============================================================
# Ch05: 正则化（sklearn 1.9 的 coef_ 一维化 + matplotlib grid）
# ============================================================
print("[Ch05] 正则化")
patch_file("Book7_Ch05_Python_Codes/Bk7_Ch05_01.ipynb",
           [("stock_levels_df = yf.download(tickers, start='2020-07-01', end='2020-12-31')",
             "stock_levels_df = pd.read_pickle('stock_levels_df.pkl')"),
            ("errors.append(mean_squared_error(clf.coef_, \n                                     b.reshape(1,-1)))",
             "errors.append(mean_squared_error(clf.coef_.ravel(), b.ravel()))"),
            ("    b_i = clf.coef_", "    b_i = clf.coef_.reshape(1, -1)"),
            ("plt.grid(b=True, which='minor', color='0.8')",
             "plt.grid(visible=True, which='minor', color='0.8')")])
patch_file("Book7_Ch05_Python_Codes/Bk1_Ch30_04.ipynb",
           [("""    data_ = np.column_stack([X,y_poly_pred])
    # 绘制散点图
    ax.scatter(X, y, s=20)
    ax.scatter(X, y_poly_pred, marker = 'x', color='k')
    # 绘制残差
    ax.plot(([i for (i,j) in data_], [i for (i,j) in data]),
            ([j for (i,j) in data_], [j for (i,j) in data]),
             c=[0.6,0.6,0.6], alpha = 0.5)""",
             """    # 绘制散点图
    ax.scatter(X, y, s=20)
    ax.scatter(X, y_poly_pred, marker = 'x', color='k')
    # 绘制残差(竖线: 观测点 -> 预测点)
    Xr = np.asarray(X).ravel()
    yr = np.asarray(y).ravel()
    ax.plot([Xr, Xr], [yr, y_poly_pred.ravel()], c=[0.6,0.6,0.6], alpha=0.5)"""),
            ("    coef = ridge.coef_[0]; # print(coef)",
             "    coef = ridge.coef_.ravel()  # sklearn1.9 单输出 coef_ 一维"),
            ("    coefs.append(ridge.coef_[0])",
             "    coefs.append(ridge.coef_)  # sklearn1.9 coef_ 一维")])

# ============================================================
# Ch06: 贝叶斯回归（pymc3 -> pymc6, 已弃API适配）
# ============================================================
print("[Ch06] 贝叶斯回归(pymc)")
f = "Book7_Ch06_Python_Codes/Bk7_Ch06_01.ipynb"
nb = read_nb(f)
subs6 = [
    ("import pymc3 as pm", "import pymc as pm"),
    ("pm.traceplot(", "import arviz as az\naz.plot_trace("),
    ("trace['alpha']", "_alpha"), ("trace['beta']", "_beta"), ("trace['sigma']", "_sigma"),
    ("trace.alpha", "_alpha"), ("trace.beta", "_beta"), ("trace.sigma", "_sigma"),
]
patch_cells(nb, subs6)
# 采样后插入扁平后验数组
for c in nb["cells"]:
    if c["cell_type"] == "code" and "pm.sample(" in "".join(c["source"]):
        src = "".join(c["source"])
        anchor = "    trace = pm.sample(draws=1000, chains=2, tune=200, \n                      discard_tuned_samples=True)"
        add = "\n\n# pymc6: InferenceData -> 扁平 numpy 数组(兼容原 trace['x'] 语义)\nimport numpy as _np\n_alpha = trace.posterior['alpha'].values.ravel()\n_beta  = trace.posterior['beta'].values.ravel()\n_sigma = trace.posterior['sigma'].values.ravel()\nn_draws = len(_alpha)"
        if anchor in src:
            c["source"] = src.replace(anchor, anchor + add).splitlines(True)
# 用 matplotlib 直方图/KDE 替换已被 arviz 移除的 plot_posterior
hist = """# Posterior hist (等价实现: matplotlib 绘制扁平后验)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for axx, name, vals in zip(axes, ['alpha','beta','sigma'], [_alpha,_beta,_sigma]):
    axx.hist(vals, bins=40, edgecolor='k', density=True, alpha=0.7)
    axx.set_title(name); axx.grid(True)
# plt.savefig('Bayesian regression hist.svg')"""
kde = """# Posterior KDE (等价实现)
import seaborn as sns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for axx, name, vals in zip(axes, ['alpha','beta','sigma'], [_alpha,_beta,_sigma]):
    sns.kdeplot(vals, ax=axx, fill=True)
    axx.set_title(name); axx.grid(True)
# plt.savefig('Bayesian regression KDE.svg')"""
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "plot_posterior(trace, kind=\"hist\"" in src:
            nb["cells"][i]["source"] = hist.splitlines(True)
        elif "plot_posterior(trace, kind=\"kde\"" in src:
            nb["cells"][i]["source"] = kde.splitlines(True)
write_nb(f, nb)
print("  [ok] Ch06 pymc 适配")

# ============================================================
# Ch07: 高斯过程（matplotlib 3D 轴 API）
# ============================================================
print("[Ch07] 高斯过程")
patch_file("Book7_Ch07_Python_Codes/Bk7_Ch07_01.ipynb",
           [("ax.w_xaxis", "ax.xaxis"), ("ax.w_yaxis", "ax.yaxis"), ("ax.w_zaxis", "ax.zaxis")])

# ============================================================
# Ch11: SVM（matplotlib cm.get_cmap / QuadContourSet）
# ============================================================
print("[Ch11] 支持向量机")
patch_file("Book7_Ch11_Python_Codes/Bk7_Ch11_01.ipynb",
           [("cm.get_cmap(", "plt.get_cmap("),
            ("for c in cnt.collections:\n        c.set_edgecolor(\"face\")",
             "cnt.set_edgecolor(\"face\")")])

# ============================================================
# Ch15: 截断型SVD（原书 cell 顺序 bug）
# ============================================================
print("[Ch15] 截断型SVD")
patch_file("Book7_Ch15_Python_Codes/Bk7_Ch15_01.ipynb",
           [("U[:, order].shape", "U[:, 0].shape  # 展示单个奇异向量形状")])

# ============================================================
# Ch16: PCA进阶（协方差 eig -> eigh 取实部）
# ============================================================
print("[Ch16] PCA进阶")
patch_file("Book7_Ch16_Python_Codes/Bk7_Ch16_01.ipynb",
           [("        variance_V, V = np.linalg.eig(GG)",
             "        variance_V, V = np.linalg.eigh(GG)  # 实对称协方差用eigh\n        variance_V = variance_V[::-1]  # 降序\n        V = V[:, ::-1]")])

# ============================================================
# Ch17: PCA与回归（联网下载 -> 本地pkl; 移除残缺补丁cell; scatter转numpy）
# ============================================================
print("[Ch17] PCA与回归")
patch_file("Book7_Ch17_Python_Codes/Bk7_Ch17_01.ipynb",
           [("""y_levels_df = yf.download(tickers = ['AAPL'],
                  start = startdate,
                  end = enddate)""",
             "y_levels_df = pd.read_pickle('y_levels_df.pkl')"),
            ("""x_levels_df = yf.download(tickers = ['^GSPC'],
                  start = startdate,
                  end = enddate)""",
             "stock_levels_df = pd.read_pickle('stock_levels_df.pkl')\nx_levels_df = stock_levels_df.xs('SP500', axis=1, level=1).sort_index()  # 该pkl中^GSPC已重命名为SP500")])
patch_file("Book7_Ch17_Python_Codes/Bk7_Ch17_02.ipynb",
           [("""X_y_df = yf.download(['AAPL','MCD','^GSPC'], start='2020-01-01', end='2020-12-31')
X_y_df.to_pickle('X_y_df.pkl')""",
             "X_y_df = pd.read_pickle('X_y_df.pkl')"),
            ("ax.scatter(X_df[\"AAPL\"], X_df[\"MCD\"], y_df,",
             "ax.scatter(X_df[\"AAPL\"].to_numpy(), X_df[\"MCD\"].to_numpy(), y_df.to_numpy(),")])
# Ch17_02 移除残缺补丁 cell（不兼容 matplotlib 3.11）
f = "Book7_Ch17_Python_Codes/Bk7_Ch17_02.ipynb"
nb = read_nb(f)
nb["cells"] = [c for c in nb["cells"]
               if "###patch start###" not in "".join(c.get("source", []))
               and "mpl_toolkits.mplot3d.axis3d import Axis" not in "".join(c.get("source", []))]
write_nb(f, nb)
print("  [ok] Ch17_02 移除残缺补丁cell")
for f in ["Bk7_Ch17_03.ipynb", "Bk7_Ch17_04.ipynb"]:
    patch_file(f"Book7_Ch17_Python_Codes/{f}",
               [("stock_levels_df = yf.download(tickers, start='2020-01-01', end='2020-12-31')",
                 "stock_levels_df = pd.read_pickle('stock_levels_df.pkl')")])

# ============================================================
# Ch19: CCA（corr 遇字符串列; eig 复数取实部）
# ============================================================
print("[Ch19] 典型相关分析")
patch_file("Book7_Ch19_Python_Codes/Bk7_Ch19_01.ipynb",
           [("iris_sns.corr()", "iris_sns.corr(numeric_only=True)"),
            ("""Lambda_U, U = np.linalg.eig(P)
Lambda_U = np.diag(Lambda_U)""",
             """Lambda_U, U = np.linalg.eig(P)
U = np.real(U)  # 复数部分为数值噪声, 取实部供可视化
Lambda_U = np.diag(np.real(Lambda_U))"""),
            ("""Lambda_V, V = np.linalg.eig(Q)
Lambda_V = np.diag(Lambda_V)""",
             """Lambda_V, V = np.linalg.eig(Q)
V = np.real(V)  # 同上取实部
Lambda_V = np.diag(np.real(Lambda_V))""")])

# ============================================================
# Ch20_03: k均值（yellowbrick 1.5 与 sklearn 1.9 不兼容 -> 手绘轮廓图）
# ============================================================
print("[Ch20_03] k均值(yellowbrick替换)")
patch_file("Book7_Ch20_Python_Codes/Bk7_Ch20_03.ipynb",
           [("""    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    
    visualizer.fit(X)
    # Fit the data to the visualizer
    visualizer.show()
    # Finalize and render the figure""",
             """    # 手绘silhouette图(替换yellowbrick, 因yellowbrick1.5与sklearn1.9不兼容)
    from sklearn.metrics import silhouette_samples
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0,
                         ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax.axvline(x=silhouette_avg, color=\"red\", linestyle=\"--\")
    ax.set_title(\"Silhouette plot, n_clusters=%d, avg=%.3f\" % (n_clusters, silhouette_avg))
    ax.set_xlabel(\"Silhouette coefficient values\")
    ax.set_ylabel(\"Cluster label\")""")])

# ============================================================
# Ch21: GMM（matplotlib Ellipse 的 angle 需关键字）
# ============================================================
print("[Ch21] 高斯混合模型")
patch_file("Book7_Ch21_Python_Codes/Bk7_Ch21_01.ipynb",
           [("""ell = Ellipse(gmm.means_[j, :2], 
                          scale*major,
                          scale*minor, 
                          angle, """,
             """ell = Ellipse(gmm.means_[j, :2], 
                          scale*major,
                          scale*minor, 
                          angle=angle,  # matplotlib新版本angle需关键字
                          """)])

print("\n完成。修补cell数合计:", report["patched"], "| 跳过:", report["skipped"])
print("提示: 全部依赖安装:  pip install -r requirements-book7.txt")
#（注：内容由AI生成）
