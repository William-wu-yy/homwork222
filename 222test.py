import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# 设置风格
sns.set(style="whitegrid")

# 读取数据
df = pd.read_csv('C:/Users/WYY/Desktop/EE6222ASS/Swarm_Behaviour.csv')

# 数据对齐和预处理
# 1. 处理缺失值
df.dropna(inplace=True)  # 丢弃含有缺失值的行

# 2. 对目标变量进行编码（如果是分类问题）
# 如果 Swarm_Behaviour 是分类类型，确保其为整数
df["Swarm_Behaviour"] = df["Swarm_Behaviour"].astype(int)

# 3. 特征和目标变量
X = df.drop("Swarm_Behaviour", axis=1)
y = df["Swarm_Behaviour"]

# 4. 数据标准化
# 使用 StandardScaler 对数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 绘制 Swarm_Behaviour 的分布图
plt.figure(figsize=(10, 6))
sns.countplot(x="Swarm_Behaviour", data=df, hue="Swarm_Behaviour", palette="coolwarm", legend=False)
plt.xticks(rotation=0)
plt.xlabel('Swarm_Behaviour')
plt.ylabel('Count')
plt.title('Swarm_Behaviour Distribution')
plt.show()

# 打印类别分布
print("Swarm_Behaviour 分布:\n", df["Swarm_Behaviour"].value_counts(normalize=True))

# PCA 计算不同主成分的方差比例
p_values = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1300, 1500, 1700, 1900, 2000]
variance = []

for n in p_values:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    variance.append(np.sum(pca.explained_variance_ratio_))

# 绘制 PCA 解释方差图
plt.figure(figsize=(12, 8))
plt.plot(p_values, variance, marker='o', linestyle='--', color='b', linewidth=2, markersize=8, label='Explained Variance')
plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
plt.xlabel("Number of Components", fontsize=14)
plt.ylabel("Explained Variance", fontsize=14)
plt.title("PCA Explained Variance vs Components", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)


plt.annotate(f'Max: {max(variance):.2f}', xy=(p_values[np.argmax(variance)], max(variance)),
             xytext=(p_values[np.argmax(variance)] + 100, max(variance) - 0.05),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='red')
plt.show()

# 使用 PCA 进行降维
pca = PCA(n_components=400)
X_pca = pca.fit_transform(X_scaled)
print("PCA Explained Variance (400 components):", np.sum(pca.explained_variance_ratio_))


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=101)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=101)


# 设置 K 值搜索范围
param_grid = {'n_neighbors': list(range(1, 15))}

# 使用 GridSearchCV 寻找最优的 K 值
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"Best K value for KNN: {best_k}")

# 使用最优的 K 值训练和评估 KNN 模型
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)
acc_knn_no_pca = accuracy_score(y_test, y_pred_knn)
print(f"KNN without PCA Accuracy (Best K={best_k}): {acc_knn_no_pca:.4f}")

# 绘制混淆矩阵（不使用PCA）
fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(knn_best, X_test, y_test, ax=ax, cmap='Blues')
plt.title("KNN (No PCA) - Confusion Matrix", fontsize=16)
plt.show()

# 绘制ROC曲线（不使用PCA）
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(knn_best, X_test, y_test, ax=ax)
plt.title("KNN (No PCA) - ROC Curve", fontsize=16)
plt.show()


# 使用 PCA 进行 KNN 模型的交叉验证
grid_search_pca = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_pca.fit(X_train_pca, y_train_pca)

best_k_pca = grid_search_pca.best_params_['n_neighbors']
print(f"Best K value for KNN with PCA: {best_k_pca}")

# 使用最优 K 值训练和评估使用 PCA 的 KNN 模型
knn_best_pca = KNeighborsClassifier(n_neighbors=best_k_pca)
knn_best_pca.fit(X_train_pca, y_train_pca)
y_pred_knn_pca = knn_best_pca.predict(X_test_pca)
acc_knn_pca = accuracy_score(y_test_pca, y_pred_knn_pca)
print(f"KNN with PCA Accuracy (Best K={best_k_pca}): {acc_knn_pca:.4f}")

# 绘制混淆矩阵（使用PCA）
fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(knn_best_pca, X_test_pca, y_test_pca, ax=ax, cmap='Blues')
plt.title("KNN (PCA) - Confusion Matrix", fontsize=16)
plt.show()

# 绘制ROC曲线（使用PCA）
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(knn_best_pca, X_test_pca, y_test_pca, ax=ax)
plt.title("KNN (PCA) - ROC Curve", fontsize=16)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 设置风格
sns.set(style="whitegrid")

# 读取数据
df = pd.read_csv('C:/Users/WYY/Desktop/EE6222ASS/Swarm_Behaviour.csv')

# 数据对齐和预处理
df.dropna(inplace=True)
df["Swarm_Behaviour"] = df["Swarm_Behaviour"].astype(int)

X = df.drop("Swarm_Behaviour", axis=1)
y = df["Swarm_Behaviour"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=400)
X_pca = pca.fit_transform(X_scaled)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=101)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=101)

# ==================== 随机森林模型（不使用PCA） ==================== #
RFC_no_pca = RandomForestClassifier(random_state=101)
params_no_pca = {
    "n_estimators": [30, 64, 100, 128, 200],
    "max_features": [2, 3, 4],
    "max_depth": [4, 8, 10, 15]
}

# 使用GridSearchCV进行调优
grid_model_no_pca = GridSearchCV(RFC_no_pca, params_no_pca, return_train_score=True, verbose=2)
grid_model_no_pca.fit(X_train, y_train)

# 将交叉验证结果转为DataFrame
cv_results_no_pca = pd.DataFrame(grid_model_no_pca.cv_results_)

# 绘制参数影响图（不使用PCA）
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
sns.lineplot(data=cv_results_no_pca, x="param_max_features", y="mean_train_score", marker='o', linestyle='--', color='b', label="Train")
sns.lineplot(data=cv_results_no_pca, x="param_max_features", y="mean_test_score", marker='o', linestyle='-', color='r', label="Test")
plt.xlabel("Max Features", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Max Features vs Score (No PCA)", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(3, 1, 2)
sns.lineplot(data=cv_results_no_pca, x="param_max_depth", y="mean_train_score", marker='o', linestyle='--', color='b', label="Train")
sns.lineplot(data=cv_results_no_pca, x="param_max_depth", y="mean_test_score", marker='o', linestyle='-', color='r', label="Test")
plt.xlabel("Max Depth", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Max Depth vs Score (No PCA)", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(3, 1, 3)
sns.lineplot(data=cv_results_no_pca, x="param_n_estimators", y="mean_train_score", marker='o', linestyle='--', color='b', label="Train")
sns.lineplot(data=cv_results_no_pca, x="param_n_estimators", y="mean_test_score", marker='o', linestyle='-', color='r', label="Test")
plt.xlabel("N Estimators", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("N Estimators vs Score (No PCA)", fontsize=14)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

# 使用最佳参数训练模型（不使用PCA）
best_params_no_pca = grid_model_no_pca.best_params_
RFC_no_pca = RandomForestClassifier(random_state=101, **best_params_no_pca)
RFC_no_pca.fit(X_train, y_train)

# 预测
y_pred_test_no_pca = RFC_no_pca.predict(X_test)
y_pred_train_no_pca = RFC_no_pca.predict(X_train)

# 计算随机森林模型不使用PCA的准确率
acc_rfc_no_pca = accuracy_score(y_test, y_pred_test_no_pca)

# 输出准确率和混淆矩阵
print(f"Best Params (No PCA): {best_params_no_pca}")
print("Train Accuracy (No PCA):", accuracy_score(y_train, y_pred_train_no_pca))
print("Test Accuracy (No PCA):", accuracy_score(y_test, y_pred_test_no_pca))

fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(RFC_no_pca, X_test, y_test, ax=ax, cmap='Blues')
plt.title("Random Forest (No PCA) - Confusion Matrix", fontsize=16)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(RFC_no_pca, X_test, y_test, ax=ax)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.title("Random Forest (No PCA) - ROC Curve", fontsize=16)
plt.show()

# ==================== 随机森林模型（使用PCA） ==================== #
RFC_pca = RandomForestClassifier(random_state=101)
params_pca = {
    "n_estimators": [30, 64, 100, 128, 200],
    "max_features": [2, 3, 4],
    "max_depth": [4, 8, 10, 15]
}

# 使用GridSearchCV进行调优
grid_model_pca = GridSearchCV(RFC_pca, params_pca, return_train_score=True, verbose=2)
grid_model_pca.fit(X_train_pca, y_train_pca)

# 将交叉验证结果转为DataFrame
cv_results_pca = pd.DataFrame(grid_model_pca.cv_results_)

# 绘制参数影响图（使用PCA）
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
sns.lineplot(data=cv_results_pca, x="param_max_features", y="mean_train_score", marker='o', linestyle='--', color='b', label="Train")
sns.lineplot(data=cv_results_pca, x="param_max_features", y="mean_test_score", marker='o', linestyle='-', color='r', label="Test")
plt.xlabel("Max Features", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Max Features vs Score (PCA)", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(3, 1, 2)
sns.lineplot(data=cv_results_pca, x="param_max_depth", y="mean_train_score", marker='o', linestyle='--', color='b', label="Train")
sns.lineplot(data=cv_results_pca, x="param_max_depth", y="mean_test_score", marker='o', linestyle='-', color='r', label="Test")
plt.xlabel("Max Depth", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Max Depth vs Score (PCA)", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(3, 1, 3)
sns.lineplot(data=cv_results_pca, x="param_n_estimators", y="mean_train_score", marker='o', linestyle='--', color='b', label="Train")
sns.lineplot(data=cv_results_pca, x="param_n_estimators", y="mean_test_score", marker='o', linestyle='-', color='r', label="Test")
plt.xlabel("N Estimators", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("N Estimators vs Score (PCA)", fontsize=14)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

# 使用最佳参数训练模型（使用PCA）
best_params_pca = grid_model_pca.best_params_
RFC_pca = RandomForestClassifier(random_state=101, **best_params_pca)
RFC_pca.fit(X_train_pca, y_train_pca)

# 预测
y_pred_test_pca = RFC_pca.predict(X_test_pca)
y_pred_train_pca = RFC_pca.predict(X_train_pca)

# 计算随机森林模型使用PCA的准确率
acc_rfc_pca = accuracy_score(y_test_pca, y_pred_test_pca)

# 输出准确率和混淆矩阵
print(f"Best Params (PCA): {best_params_pca}")
print("Train Accuracy (PCA):", accuracy_score(y_train_pca, y_pred_train_pca))
print("Test Accuracy (PCA):", accuracy_score(y_test_pca, y_pred_test_pca))

fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(RFC_pca, X_test_pca, y_test_pca, ax=ax, cmap='Blues')
plt.title("Random Forest (PCA) - Confusion Matrix", fontsize=16)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(RFC_pca, X_test_pca, y_test_pca, ax=ax)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.title("Random Forest (PCA) - ROC Curve")
plt.show()


# 绘制准确率对比图
labels = ['KNN (No PCA)', 'KNN (PCA)', 'RF (No PCA)', 'RF (PCA)']

accuracies = [acc_knn_no_pca, acc_knn_pca, acc_rfc_no_pca, acc_rfc_pca]


print("acc_knn_no_pca:", acc_knn_no_pca)
print("acc_knn_pca:", acc_knn_pca)
print("acc_rfc_no_pca:", acc_rfc_no_pca)
print("acc_rfc_pca:", acc_rfc_pca)



plt.figure(figsize=(3.5, 2.5), dpi=300)
sns.barplot(x=labels, y=accuracies, palette="coolwarm")
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Comparison (With/Without PCA)', fontsize=12)
plt.ylim(0.8, 1)
plt.xticks(rotation=0, fontsize=8)  # 将rotation设置为0，保持水平
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()
