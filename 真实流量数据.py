import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.ensemble import IsolationForest
from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde, entropy

# 局部线性核回归（基于GCV的带宽选择）
def local_linear_regression(x, y):
    """优化带宽选择并返回残差"""

    def gcv_error(h):
        try:
            kr = KernelReg(y, x, var_type='c', reg_type='ll', bw=[h])  # bw 应该是列表
            y_pred, _ = kr.fit(x)
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            return mse
        except Exception as e:
            print(f"Error in gcv_error with bandwidth {h}: {e}")
            return np.inf

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
    mad = np.mean(np.abs(data - median))
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

def calculate_neighbor_difference(data, k=1):
    """
    计算数据点与其前后 k 个邻居点的平均差异，去除自身影响
    """
    n = len(data)
    differences = np.zeros(n)

    for i in range(n):
        neighbors = []
        for j in range(i - k, i + k + 1):
            if 0 <= j < n and j != i:  # 确保索引有效且不包含自身
                neighbors.append(data[j])

        if neighbors:  # 避免空列表求均值报错
            differences[i] = np.mean(np.abs(data[i] - np.array(neighbors)))

    return differences


def entropy_weight(r_diff, d_diff):
    """
    计算基于信息熵的自适应权重
    """
    # 计算熵
    H_r = entropy(np.histogram(r_diff, bins=1000, density=True)[0] + 1e-6)  # 避免log(0)
    H_d = entropy(np.histogram(d_diff, bins=1000, density=True)[0] + 1e-6)

    total_entropy = H_r + H_d
    if total_entropy == 0:
        return 0.5, 0.5  # 避免除零，默认均分

    w1 = H_r / total_entropy
    w2 = H_d / total_entropy
    return w1, w2


def anomaly_score(residuals, y, k=10):
    """
    结合残差和密度生成动态异常分数，使用基于熵的自适应权重
    """
    # 残差归一化
    r_norm = mad_normalize(weight_r(residuals, y))

    # 计算核密度估计（KDE）并归一化
    kde = gaussian_kde(y.reshape(-1))
    density = kde(y.reshape(-1))
    d_norm = mad_normalize(density)

    # 计算残差和密度的邻居差异
    r_neighbor_diff = calculate_neighbor_difference(r_norm, k)
    d_neighbor_diff = calculate_neighbor_difference(d_norm, k)

    # 计算基于熵的自适应权重
    w1, w2 = entropy_weight(r_neighbor_diff, d_neighbor_diff)
    # 计算最终异常分数
    return w1 * r_norm + w2 * (1 - d_norm)


# 1. 数据读取与预处理
df = pd.read_csv("618KS001.csv", parse_dates=["TM"])
df = df[["TM", "INQ"]].dropna().sort_values("TM").reset_index(drop=True)

# 处理缺失值（线性插值）
df["INQ"] = df["INQ"].interpolate()

# 2. 局部线性核回归拟合
# 转换为数值型时间戳（用于回归）
time_numeric = (df["TM"] - df["TM"].min()).dt.total_seconds().values.reshape(-1, 1)
y = df["INQ"].values

# 执行局部线性核回归
residuals, h_opt = local_linear_regression(time_numeric, y)
print(f"Optimal bandwidth: {h_opt}")

# 3. 计算异常分数
anomaly_scores = anomaly_score(residuals, y)

# 计算数据的均值
mean_value = np.median(anomaly_scores)
# 计算数据的标准差
std_value = np.mean(np.abs(anomaly_scores - mean_value))

# 计算异常值的下限（均值 - 3倍标准差）
lower_threshold = mean_value - 3 * std_value
# 计算异常值的上限（均值 + 3倍标准差）
upper_threshold = mean_value + 3 * std_value

anomalies = ((anomaly_scores > upper_threshold) | (anomaly_scores < lower_threshold))

# 4. 对比分析：使用Isolation Forest进行异常检测
clf = IsolationForest(contamination=0.05)
clf.fit(time_numeric)
anomalies_isolation_forest = clf.predict(time_numeric) == -1

# 5. 结果可视化
plt.figure(figsize=(12, 6))
# 使用更符合科研规范的颜色
plt.plot(df["TM"], df["INQ"], color='dodgerblue', label="Flow (INQ)")
plt.scatter(df["TM"][anomalies], df["INQ"][anomalies], color='tomato', label="Anomalies (Our Method)")
plt.scatter(df["TM"][anomalies_isolation_forest], df["INQ"][anomalies_isolation_forest], color='mediumseagreen', label="Anomalies (Isolation Forest)")
plt.title("Flow Data with Anomalies Detected")
plt.xlabel("Time")
plt.ylabel("Flow")
plt.legend()
# 去除网格线
plt.grid(False)
plt.show()

# 输出异常点统计
print(f"Detected {sum(anomalies)} anomalies using Our method.")
print(f"Detected {sum(anomalies_isolation_forest)} anomalies using Isolation Forest.")

# 计算无监督评价指标
# 为两种方法的结果创建标签
labels_our_method = anomalies.astype(int)
labels_isolation_forest = anomalies_isolation_forest.astype(int)

# 计算轮廓系数
silhouette_our_method = silhouette_score(time_numeric, labels_our_method)
silhouette_isolation_forest = silhouette_score(time_numeric, labels_isolation_forest)

print(f"Silhouette score for Our method: {silhouette_our_method}")
print(f"Silhouette score for Isolation Forest: {silhouette_isolation_forest}")