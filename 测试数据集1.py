import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize_scalar
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
# =========================生成模拟数据==================
# 生成加性异常点（AO）
def generate_ao(y, n, contamination, seed):
    num_ao = int(n * contamination * 0.2)  # AO占20%异常
    np.random.seed(seed + 1)  # 为AO生成设置不同的随机种子
    ao_indices = np.random.choice(n, num_ao, replace=False)
    y[ao_indices] += np.random.choice([-20, 20], num_ao)
    return y, ao_indices


# 生成革新异常点（IO）
def generate_io(y, n, contamination, seed):
    num_io = int(n * contamination * 0.2)  # IO占20%异常
    if num_io > 0:
        np.random.seed(seed + 2)  # 为IO生成设置不同的随机种子
        io_start = np.random.randint(int(n * 0.6), int(n * 0.9))  # 随机起始位置
        io_length = max(10, num_io)  # 最小长度10
        y[io_start:io_start + io_length] += 10 + np.random.randn(io_length) * 2.5
        return y, io_start, io_length
    return y, None, None


# 生成周期性异常点
def generate_periodic(y, n, contamination, seed):
    num_periodic = int(n * contamination * 0.3)  # 周期性异常点占30%异常
    if num_periodic > 0:
        np.random.seed(seed + 3)
        period = np.random.randint(5, 20)  # 随机周期
        periodic_indices = np.arange(0, n, period)[:num_periodic]
        y[periodic_indices] += np.random.normal(10, 0.5, num_periodic)
        return y, periodic_indices
    return y, None


# 生成趋势异常点
def generate_trend(y, n, contamination, seed):
    num_trend = int(n * contamination * 0.3)  # 趋势异常点占30%异常
    if num_trend > 0:
        np.random.seed(seed + 4)
        trend_start = np.random.randint(0, n - num_trend)
        trend_slope = np.random.choice([-5, 5])
        trend_anomaly = np.arange(num_trend) * trend_slope + np.random.normal(0, 0.2, num_trend)
        y[trend_start:trend_start + num_trend] += trend_anomaly
        return y, trend_start, num_trend
    return y, None, None


# 生成模拟数据
def generate_data(n=1000, contamination=0.1, seed=2):
    np.random.seed(seed)
    x = np.linspace(-10, 10, n)
    # 正常数据：增加复杂度，包含正弦、余弦和多项式函数
    y = 1.5 * np.sin(7 * x + np.pi / 3) + 0.3 * np.cos(9 * x - np.pi / 8) \
        + np.random.normal(0, 0.1, n)

    # 生成各种异常点
    y, ao_indices = generate_ao(y, n, contamination, seed)
    y, io_start, io_length = generate_io(y, n, contamination, seed)
    y, periodic_indices = generate_periodic(y, n, contamination, seed)
    y, trend_start, trend_length = generate_trend(y, n, contamination, seed)

    # 生成标签（AO、IO、周期性异常点和趋势异常点合并）
    labels = np.zeros(n)
    labels[ao_indices] = 1
    if io_start is not None:
        labels[io_start:io_start + io_length] = 1
    if periodic_indices is not None:
        labels[periodic_indices] = 1
    if trend_start is not None:
        labels[trend_start:trend_start + trend_length] = 1
    return x, y, labels

# ================  局部线性核回归 =========================
# 2. 局部线性核回归（基于GCV的带宽选择）
def local_linear_regression(x, y):
    """优化带宽选择并返回残差"""
    def gcv_error(h):
        kr = KernelReg(y, x, var_type='c', reg_type='ll', bw=[h])  # bw 应该是列表
        y_pred, _ = kr.fit(x)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
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


#=====================权重=========================
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

    # normal_residual = residuals[normal_mask] * 1
    anomaly_residual = residuals[anomaly_mask] * anomaly_weight

    # 合并正常和异常残差
    weighted_residual = np.zeros_like(residuals)
    # weighted_residual[normal_mask] = normal_residual
    weighted_residual[anomaly_mask] = anomaly_residual


    return weighted_residual


def anomaly_score(residuals, y):
    """
    结合残差和密度生成动态异常分数，使用基于熵的自适应权重
    """
    weighted_r = weight_r(residuals, y)
    # 残差归一化
    r_norm = mad_normalize(weighted_r)

    # 计算核密度估计（KDE）并归一化
    kde = gaussian_kde(y.reshape(-1))
    density = kde(y.reshape(-1))
    d_norm = mad_normalize(density)

    # 计算基于熵的自适应权重
    w1 = r_norm/ (r_norm + d_norm)
    w2 = 1 - w1

    # 计算最终异常分数
    return  w1 * r_norm + w2 * (1 - d_norm)


#=======================对比试验++++++++++++++++++++++++
# PCA 异常检测
def pca_anomaly_detection(y, contamination):
    pca = PCA(n_components=1)
    y_pca = pca.fit_transform(y.reshape(-1, 1))
    reconstructions = pca.inverse_transform(y_pca)
    mse = np.mean(np.power(y.reshape(-1, 1) - reconstructions, 2), axis=1)
    threshold = np.quantile(mse, 1 - contamination)
    return (mse > threshold).astype(int)

# 局部线性核回归异常检测
def local_linear_regression_anomaly_detection(x, y, contamination):
    residuals, h_opt = local_linear_regression(x, y)
    upper_quantile = 1 - contamination / 2
    lower_quantile = contamination / 2
    upper_threshold = np.quantile(residuals, upper_quantile)
    lower_threshold = np.quantile(residuals, lower_quantile)
    # upper_threshold = np.quantile(residuals, upper_quantile)
    return ((residuals > upper_threshold) | (residuals < lower_threshold)).astype(int)

# 定义不同的污染度
contaminations = [0.01, 0.03, 0.05, 0.10]

for contamination in contaminations:
    print(f"\nRunning experiment with contamination = {contamination}")
    # 5. 运行实验
    x, y, true_labels = generate_data(n=500, contamination=contamination)
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

    correctly_detected_custom = np.sum((pred_labels_custom == 1) & (true_labels == 1))
    num_true_anomalies = np.sum(true_labels)
    num_detected_custom = np.sum(pred_labels_custom)

    # 局部线性核回归方法
    pred_labels_llr = local_linear_regression_anomaly_detection(x, y, contamination)
    correctly_detected_llr = np.sum((pred_labels_llr == 1) & (true_labels == 1))
    num_detected_llr = np.sum(pred_labels_llr)

    # Isolation Forest
    clf_if = IsolationForest(contamination=contamination)
    pred_labels_if = clf_if.fit_predict(y.reshape(-1, 1))
    pred_labels_if = (pred_labels_if == -1).astype(int)
    correctly_detected_if = np.sum((pred_labels_if == 1) & (true_labels == 1))
    num_detected_if = np.sum(pred_labels_if)

    # Local Outlier Factor
    clf_lof = LocalOutlierFactor(contamination=contamination)
    pred_labels_lof = clf_lof.fit_predict(y.reshape(-1, 1))
    pred_labels_lof = (pred_labels_lof == -1).astype(int)
    correctly_detected_lof = np.sum((pred_labels_lof == 1) & (true_labels == 1))
    num_detected_lof = np.sum(pred_labels_lof)

    # One-Class SVM
    clf_ocsvm = OneClassSVM(nu=contamination)
    pred_labels_ocsvm = clf_ocsvm.fit_predict(y.reshape(-1, 1))
    pred_labels_ocsvm = (pred_labels_ocsvm == -1).astype(int)
    correctly_detected_ocsvm = np.sum((pred_labels_ocsvm == 1) & (true_labels == 1))
    num_detected_ocsvm = np.sum(pred_labels_ocsvm)

    # DBSCAN
    db = DBSCAN(eps=0.5, min_samples=3).fit(y.reshape(-1, 1))
    pred_labels_dbscan = (db.labels_ == -1).astype(int)
    correctly_detected_dbscan = np.sum((pred_labels_dbscan == 1) & (true_labels == 1))
    num_detected_dbscan = np.sum(pred_labels_dbscan)

    # PCA
    pred_labels_pca = pca_anomaly_detection(y, contamination)
    correctly_detected_pca = np.sum((pred_labels_pca == 1) & (true_labels == 1))
    num_detected_pca = np.sum(pred_labels_pca)

    # 输出对比结果
    print(f"Total True Anomalies: {num_true_anomalies}")
    print("Custom Method:")
    print(f"  Detected Anomalies: {num_detected_custom}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_custom}")
    print("Local Linear Regression:")
    print(f"  Detected Anomalies: {num_detected_llr}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_llr}")
    print("Isolation Forest:")
    print(f"  Detected Anomalies: {num_detected_if}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_if}")
    print("Local Outlier Factor:")
    print(f"  Detected Anomalies: {num_detected_lof}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_lof}")
    print("One-Class SVM:")
    print(f"  Detected Anomalies: {num_detected_ocsvm}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_ocsvm}")
    print("DBSCAN:")
    print(f"  Detected Anomalies: {num_detected_dbscan}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_dbscan}")
    print("PCA:")
    print(f"  Detected Anomalies: {num_detected_pca}")
    print(f"  Correctly Detected Anomalies: {correctly_detected_pca}")

    # 6. 评估性能并可视化
    # 绘制生成的数据和真实异常点
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Data', color='blue')
    plt.scatter(x[true_labels == 1], y[true_labels == 1], color='red', label='True Outliers')
    plt.legend()
    plt.title(f"Generated Data with True Anomalies (Contamination = {contamination})")
    plt.grid(False)  # 关闭网格线
    plt.savefig(f"generated_data_contamination_{contamination}.png", dpi=300)
    plt.close()

    # 绘制局部线性核回归异常分数和检测到的异常点
    residuals, h_opt = local_linear_regression(x, y)
    scores = anomaly_score(residuals, y)
    plt.figure(figsize=(10, 6))
    plt.plot(x, scores, label='Anomaly Score', color='green')
    plt.scatter(x[true_labels == 1], scores[true_labels == 1], color='black', label='Detected Anomalies')
    plt.legend()
    plt.title(f"Anomaly Scores with Detected Anomalies (Contamination = {contamination})")
    plt.grid(False)  # 关闭网格线
    plt.savefig(f"anomaly_scores_contamination_{contamination}.png", dpi=300)
    plt.close()

    # 可视化不同方法的检测结果
    methods = ['Ours', 'Local Linear Regression', 'Isolation Forest', 'Local Outlier Factor', 'One - Class SVM', 'PCA']
    pred_labels = [pred_labels_custom, pred_labels_llr, pred_labels_if, pred_labels_lof, pred_labels_ocsvm, pred_labels_pca]
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(methods))
    for i, method in enumerate(methods):
        if method == 'Ours':
            # 突出显示自定义方法的检测点
            plt.scatter(x[pred_labels[i] == 1], y[pred_labels[i] == 1], label=method, color=colors[i], alpha=1, s=80, marker='*')
        else:
            plt.scatter(x[pred_labels[i] == 1], y[pred_labels[i] == 1], label=method, color=colors[i], alpha=0.5)
    plt.legend()
    plt.title(f"Anomaly Detection Results by Different Methods (Contamination = {contamination})")
    plt.grid(False)  # 关闭网格线
    plt.savefig(f"anomaly_detection_results_contamination_{contamination}.png", dpi=300)
    plt.close()
