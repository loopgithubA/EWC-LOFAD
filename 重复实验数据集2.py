import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_ind
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima_process import arma_generate_sample

# =========================生成模拟数据==================
# 1. 生成模拟数据（改进AO/IO生成逻辑）
def generate_data(n=500, contamination=0.1, seed=42):
    np.random.seed(seed)
    # 定义 ARMA 模型的参数
    ar = np.array([1, -0.5])  # AR 系数
    ma = np.array([1, 0.2])   # MA 系数
    # 生成 ARMA 时间序列数据
    y = arma_generate_sample(ar, ma, nsample=n)

    # 生成加性异常点（AO）：孤立尖峰
    num_ao = int(n * contamination * 0.5)  # AO占50%异常
    np.random.seed(seed + 1)  # 为AO生成设置不同的随机种子
    ao_indices = np.random.choice(n, num_ao, replace=False)
    y[ao_indices] += np.random.choice([-20, 20], num_ao)

    # 生成革新异常点（IO）：持续结构变化
    if contamination > 0:
        np.random.seed(seed + 2)  # 为IO生成设置不同的随机种子
        io_start = np.random.randint(int(n * 0.6), int(n * 0.9))  # 随机起始位置
        io_length = max(10, int(n * contamination * 0.5))  # 最小长度10，占50%异常
        y[io_start:io_start + io_length] += 10 + np.random.randn(io_length) * 2.5

    # 生成标签（AO和IO合并）
    labels = np.zeros(n)
    labels[ao_indices] = 1
    labels[io_start:io_start + io_length] = 1
    x = np.arange(n)  # 时间索引
    return x, y, labels


# 局部线性核回归（基于GCV的带宽选择）
def local_linear_regression(x, y):
    """优化带宽选择并返回残差"""

    def gcv_error(h):
        kr = KernelReg(y, x, var_type='c', reg_type='ll', bw=[h])  # bw 应该是列表
        y_pred, _ = kr.fit(x)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        return mse

    # 自动搜索最优带宽
    sigma = np.std(y)
    result = minimize_scalar(
        lambda h: gcv_error(h) if h > 0 else np.inf,
        bounds=(0.1 * sigma, 2 * sigma),
        method='bounded'
    )
    h_opt = result.x

    # 最终拟合
    kr_opt = KernelReg(y, x, var_type='c', reg_type='ll', bw=[h_opt])
    y_pred, _ = kr_opt.fit(x)
    return y - y_pred, h_opt


# ====================权重========================
def mad_normalize(data):
    """
    使用中位数绝对偏差 (MAD) 归一化数据
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return (data - median) / (mad + 1e-6)  # 防止除零

# 使用 K - distance 方法确定 DBSCAN 超参数
def k_distance(data, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    k_dist = np.sort(distances[:, k - 1])
    return k_dist

def find_elbow_point(k_dist):
    n = len(k_dist)
    start_point = np.array([0, k_dist[0]])
    end_point = np.array([n - 1, k_dist[-1]])
    distances = []
    for i in range(n):
        point = np.array([i, k_dist[i]])
        distance = np.linalg.norm(np.cross(end_point - start_point, start_point - point)) / np.linalg.norm(
            end_point - start_point)
        distances.append(distance)
    elbow_index = np.argmax(distances)
    return k_dist[elbow_index]

# 优化DBSCAN参数
def optimize_dbscan(y, k_range=[3, 5, 7]):
    best_score = -np.inf
    best_params = {'eps': 0.5, 'min_samples': 5}  # 默认值

    for k in k_range:
        k_dist = k_distance(y.reshape(-1, 1), k)
        eps = find_elbow_point(k_dist)

        # 缩小参数搜索范围
        param_grid = {
            'eps': [eps * 0.7, eps, eps * 1.3],
            'min_samples': [max(3, k - 2), k, min(10, k + 2)]
        }

        for params in ParameterGrid(param_grid):
            db = DBSCAN(**params).fit(y.reshape(-1, 1))
            labels = db.labels_

            if len(np.unique(labels)) > 1:
                # 使用密度聚类质量指标替代轮廓系数
                noise_ratio = np.sum(labels == -1) / len(y)
                if 0.05 < noise_ratio < 0.3:  # 限制噪声比例
                    score = silhouette_score(y.reshape(-1, 1), labels)
                    if score > best_score:
                        best_score = score
                        best_params = params

    return best_params['eps'], best_params['min_samples']

def weight_r(residuals, y, k=5):
    best_eps, best_min_samples = optimize_dbscan(y, k_range=[k])
    db = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(y.reshape(-1, 1))

    # 区分正常和异常数据
    normal_mask = db.labels_ != -1
    anomaly_mask = db.labels_ == -1

    # 计算正常和异常数据的数量
    normal_count = np.sum(normal_mask)
    anomaly_count = np.sum(anomaly_mask)
    total_points = len(y)

    # 计算权重
    anomaly_weight = total_points / anomaly_count

    anomaly_residual = residuals[anomaly_mask] * anomaly_weight

    # 合并正常和异常残差
    weighted_residual = np.zeros_like(residuals)
    weighted_residual[anomaly_mask] = anomaly_residual

    return weighted_residual


def anomaly_score(residuals, y):
    """
    结合残差和密度生成动态异常分数，使用基于熵的自适应权重
    """
    # 残差归一化
    r_norm = mad_normalize(weight_r(residuals, y))

    # 计算核密度估计（KDE）并归一化
    kde = gaussian_kde(y.reshape(-1))
    density = kde(y.reshape(-1))
    d_norm = mad_normalize(density)

    # 计算基于熵的自适应权重
    w1 = r_norm/ (r_norm + d_norm)
    w2 = 1 - w1

    # 计算最终异常分数
    return w1 * r_norm + w2 * (1 - d_norm)


# PyTorch Autoencoder 异常检测
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# PCA 异常检测
def pca_anomaly_detection(y, contamination):
    pca = PCA(n_components=1)
    y_pca = pca.fit_transform(y.reshape(-1, 1))
    reconstructions = pca.inverse_transform(y_pca)
    mse = np.mean(np.power(y.reshape(-1, 1) - reconstructions, 2), axis=1)
    threshold = np.quantile(mse, 1 - contamination)
    return (mse > threshold).astype(int)


# 评估异常检测方法
def evaluate_method(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return precision, recall, f1


# 定义不同的污染度
contaminations = [0.01, 0.03, 0.05, 0.10]
num_trials = 20

for contamination in contaminations:
    print(f"\nRunning experiment with contamination = {contamination}")
    custom_f1_scores = []
    if_f1_scores = []
    lof_f1_scores = []
    ocsvm_f1_scores = []
    dbscan_f1_scores = []
    pca_f1_scores = []
    local_linear_f1_scores = []

    custom_precision_scores = []
    if_precision_scores = []
    lof_precision_scores = []
    ocsvm_precision_scores = []
    dbscan_precision_scores = []
    pca_precision_scores = []
    local_linear_precision_scores = []

    custom_recall_scores = []
    if_recall_scores = []
    lof_recall_scores = []
    ocsvm_recall_scores = []
    dbscan_recall_scores = []
    pca_recall_scores = []
    local_linear_recall_scores = []

    for trial in range(num_trials):
        # 5. 运行实验
        x, y, true_labels = generate_data(n=500, contamination=contamination, seed=trial)

        # 自定义方法
        residuals, h_opt = local_linear_regression(x, y)
        scores = anomaly_score(residuals, y)
        # 计算数据的均值
        mean_value = np.median(scores)
        # 计算数据的标准差
        std_value = np.mean(np.abs(scores - mean_value))

        # 计算异常值的下限（均值 - 3倍标准差）
        lower_threshold = mean_value - 3 * std_value
        # 计算异常值的上限（均值 + 3倍标准差）
        upper_threshold = mean_value + 3 * std_value

        pred_labels_custom = ((scores > upper_threshold) | (scores < lower_threshold)).astype(int)
        precision_custom, recall_custom, f1_custom = evaluate_method(true_labels, pred_labels_custom)
        custom_f1_scores.append(f1_custom)
        custom_precision_scores.append(precision_custom)
        custom_recall_scores.append(recall_custom)

        # Isolation Forest
        clf_if = IsolationForest(contamination=contamination)
        pred_labels_if = clf_if.fit_predict(y.reshape(-1, 1))
        pred_labels_if = (pred_labels_if == -1).astype(int)
        precision_if, recall_if, f1_if = evaluate_method(true_labels, pred_labels_if)
        if_f1_scores.append(f1_if)
        if_precision_scores.append(precision_if)
        if_recall_scores.append(recall_if)

        # Local Outlier Factor
        clf_lof = LocalOutlierFactor(contamination=contamination)
        pred_labels_lof = clf_lof.fit_predict(y.reshape(-1, 1))
        pred_labels_lof = (pred_labels_lof == -1).astype(int)
        precision_lof, recall_lof, f1_lof = evaluate_method(true_labels, pred_labels_lof)
        lof_f1_scores.append(f1_lof)
        lof_precision_scores.append(precision_lof)
        lof_recall_scores.append(recall_lof)

        # One-Class SVM
        clf_ocsvm = OneClassSVM(nu=contamination)
        pred_labels_ocsvm = clf_ocsvm.fit_predict(y.reshape(-1, 1))
        pred_labels_ocsvm = (pred_labels_ocsvm == -1).astype(int)
        precision_ocsvm, recall_ocsvm, f1_ocsvm = evaluate_method(true_labels, pred_labels_ocsvm)
        ocsvm_f1_scores.append(f1_ocsvm)
        ocsvm_precision_scores.append(precision_ocsvm)
        ocsvm_recall_scores.append(recall_ocsvm)

        # DBSCAN
        db = DBSCAN(eps=0.5, min_samples=3).fit(y.reshape(-1, 1))
        pred_labels_dbscan = (db.labels_ == -1).astype(int)
        precision_dbscan, recall_dbscan, f1_dbscan = evaluate_method(true_labels, pred_labels_dbscan)
        dbscan_f1_scores.append(f1_dbscan)
        dbscan_precision_scores.append(precision_dbscan)
        dbscan_recall_scores.append(recall_dbscan)

        # PCA
        pred_labels_pca = pca_anomaly_detection(y, contamination)
        precision_pca, recall_pca, f1_pca = evaluate_method(true_labels, pred_labels_pca)
        pca_f1_scores.append(f1_pca)
        pca_precision_scores.append(precision_pca)
        pca_recall_scores.append(recall_pca)

        # 仅局部线性核回归方法
        residuals, _ = local_linear_regression(x, y)
        residual_scores = np.abs(residuals)
        threshold = np.quantile(residual_scores, 1 - contamination)
        pred_labels_local_linear = (residual_scores > threshold).astype(int)
        precision_local_linear, recall_local_linear, f1_local_linear = evaluate_method(true_labels, pred_labels_local_linear)
        local_linear_f1_scores.append(f1_local_linear)
        local_linear_precision_scores.append(precision_local_linear)
        local_linear_recall_scores.append(recall_local_linear)

    # 计算均值和标准差
    methods = ['Ours', 'Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'DBSCAN', 'PCA', 'Local Linear']
    f1_scores_list = [custom_f1_scores, if_f1_scores, lof_f1_scores, ocsvm_f1_scores, dbscan_f1_scores,
                       pca_f1_scores, local_linear_f1_scores]
    precision_scores_list = [custom_precision_scores, if_precision_scores, lof_precision_scores, ocsvm_precision_scores,
                             dbscan_precision_scores, pca_precision_scores, local_linear_precision_scores]
    recall_scores_list = [custom_recall_scores, if_recall_scores, lof_recall_scores, ocsvm_recall_scores,
                          dbscan_recall_scores, pca_recall_scores, local_linear_recall_scores]

    print("F1 Scores:")
    for i, method in enumerate(methods):
        mean_f1 = np.mean(f1_scores_list[i])
        std_f1 = np.std(f1_scores_list[i])
        print(f"{method}: {mean_f1:.4f} ± {std_f1:.4f}")
        if method != 'Ours':
            t_stat, p_value = ttest_ind(custom_f1_scores, f1_scores_list[i])
            print(f"  t-test vs Custom: t = {t_stat:.4f}, p = {p_value:.4f}")

    print("\nPrecision Scores:")
    for i, method in enumerate(methods):
        mean_precision = np.mean(precision_scores_list[i])
        std_precision = np.std(precision_scores_list[i])
        print(f"{method}: {mean_precision:.4f} ± {std_precision:.4f}")

    print("\nRecall Scores:")
    for i, method in enumerate(methods):
        mean_recall = np.mean(recall_scores_list[i])
        std_recall = np.std(recall_scores_list[i])
        print(f"{method}: {mean_recall:.4f} ± {std_recall:.4f}")

    # 可视化F1分数
    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=[np.mean(f1) for f1 in f1_scores_list])
    plt.errorbar(x=methods, y=[np.mean(f1) for f1 in f1_scores_list],
                 yerr=[np.std(f1) for f1 in f1_scores_list], fmt='none', ecolor='black', capsize=5)
    plt.title(f"F1 Scores Comparison (Contamination = {contamination})")
    plt.xlabel("Methods")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.savefig(f"f1_scores_contamination_{contamination}.png", dpi=300)
    plt.close()


# 可视化Precision分数
    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=[np.mean(precision) for precision in precision_scores_list])
    plt.errorbar(x=methods, y=[np.mean(precision) for precision in precision_scores_list],
                 yerr=[np.std(precision) for precision in precision_scores_list], fmt='none', ecolor='black', capsize=5)
    plt.title(f"Precision Scores Comparison (Contamination = {contamination})")
    plt.xlabel("Methods")
    plt.ylabel("Precision Score")
    plt.xticks(rotation=45)
    plt.savefig(f"precision_scores_contamination_{contamination}.png", dpi=300)
    plt.close()

    # 可视化Recall分数
    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=[np.mean(recall) for recall in recall_scores_list])
    plt.errorbar(x=methods, y=[np.mean(recall) for recall in recall_scores_list],
                 yerr=[np.std(recall) for recall in recall_scores_list], fmt='none', ecolor='black', capsize=5)
    plt.title(f"Recall Scores Comparison (Contamination = {contamination})")
    plt.xlabel("Methods")
    plt.ylabel("Recall Score")
    plt.xticks(rotation=45)
    plt.savefig(f"recall_scores_contamination_{contamination}.png", dpi=300)
    plt.close()

    # 可视化各指标的箱线图（可选拓展）
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    data_f1 = pd.DataFrame(dict(zip(methods, f1_scores_list)))
    sns.boxplot(data=data_f1)
    plt.title(f"F1 Scores Boxplot (Contamination = {contamination})")
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    data_precision = pd.DataFrame(dict(zip(methods, precision_scores_list)))
    sns.boxplot(data=data_precision)
    plt.title(f"Precision Scores Boxplot (Contamination = {contamination})")
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    data_recall = pd.DataFrame(dict(zip(methods, recall_scores_list)))
    sns.boxplot(data=data_recall)
    plt.title(f"Recall Scores Boxplot (Contamination = {contamination})")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"boxplots_contamination_{contamination}.png", dpi=300)
    plt.close()

    # 输出各方法在当前污染度下的指标总结
    print(f"\nSummary for contamination {contamination}:")
    for i, method in enumerate(methods):
        print(f"Method: {method}")
        print(f"  Mean F1: {np.mean(f1_scores_list[i]):.4f} ± {np.std(f1_scores_list[i]):.4f}")
        print(f"  Mean Precision: {np.mean(precision_scores_list[i]):.4f} ± {np.std(precision_scores_list[i]):.4f}")
        print(f"  Mean Recall: {np.mean(recall_scores_list[i]):.4f} ± {np.std(recall_scores_list[i]):.4f}")
        if method != 'Ours':
            t_stat_f1, p_value_f1 = ttest_ind(custom_f1_scores, f1_scores_list[i])
            print(f"  t-test (F1) vs Ours: t = {t_stat_f1:.4f}, p = {p_value_f1:.4f}")