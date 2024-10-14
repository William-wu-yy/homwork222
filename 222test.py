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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


sns.set(style="whitegrid")


df = pd.read_csv('C:/Users/WYY/Desktop/EE6222ASS/Swarm_Behaviour.csv')


df.dropna(inplace=True)



df["Swarm_Behaviour"] = df["Swarm_Behaviour"].astype(int)



X = df.drop("Swarm_Behaviour", axis=1)
y = df["Swarm_Behaviour"]




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



plt.figure(figsize=(10, 6))
sns.countplot(x="Swarm_Behaviour", data=df, hue="Swarm_Behaviour", palette="coolwarm", legend=False)
plt.xticks(rotation=0)
plt.xlabel('Swarm_Behaviour')
plt.ylabel('Count')
plt.title('Swarm_Behaviour Distribution')
plt.show()



print("Swarm_Behaviour 分布:\n", df["Swarm_Behaviour"].value_counts(normalize=True))



p_values = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1300, 1500, 1700, 1900, 2000]
variance = []

for n in p_values:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    variance.append(np.sum(pca.explained_variance_ratio_))



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



pca = PCA(n_components=400)
X_pca = pca.fit_transform(X_scaled)
print("PCA Explained Variance (400 components):", np.sum(pca.explained_variance_ratio_))



n_classes = len(np.unique(y))
n_features = X_scaled.shape[1]



n_components_lda = min(n_features, n_classes - 1)
lda = LDA(n_components=n_components_lda)



X_lda = lda.fit_transform(X_scaled, y)
print(f"LDA Explained Variance (n_components={n_components_lda}):", lda.explained_variance_ratio_)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=101)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=101)


X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.25, random_state=101)



param_grid = {'n_neighbors': list(range(1, 15))}


grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"Best K value for KNN: {best_k}")



knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)
acc_knn_no_pca = accuracy_score(y_test, y_pred_knn)
print(f"KNN without PCA Accuracy (Best K={best_k}): {acc_knn_no_pca:.4f}")


fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(knn_best, X_test, y_test, ax=ax, cmap='Blues')
plt.title("KNN (No PCA) - Confusion Matrix", fontsize=16)
plt.show()



fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(knn_best, X_test, y_test, ax=ax)
plt.title("KNN (No PCA) - ROC Curve", fontsize=16)
plt.show()


grid_search_pca = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_pca.fit(X_train_pca, y_train_pca)

best_k_pca = grid_search_pca.best_params_['n_neighbors']
print(f"Best K value for KNN with PCA: {best_k_pca}")


knn_best_pca = KNeighborsClassifier(n_neighbors=best_k_pca)
knn_best_pca.fit(X_train_pca, y_train_pca)
y_pred_knn_pca = knn_best_pca.predict(X_test_pca)
acc_knn_pca = accuracy_score(y_test_pca, y_pred_knn_pca)
print(f"KNN with PCA Accuracy (Best K={best_k_pca}): {acc_knn_pca:.4f}")


fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(knn_best_pca, X_test_pca, y_test_pca, ax=ax, cmap='Blues')
plt.title("KNN (PCA) - Confusion Matrix", fontsize=16)
plt.show()


fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(knn_best_pca, X_test_pca, y_test_pca, ax=ax)
plt.title("KNN (PCA) - ROC Curve", fontsize=16)
plt.show()

#-------------------------------------------------------------------

grid_search_lda = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_lda.fit(X_train_lda, y_train_lda)

best_k_lda = grid_search_lda.best_params_['n_neighbors']
print(f"Best K value for KNN with LDA: {best_k_lda}")


knn_best_lda = KNeighborsClassifier(n_neighbors=best_k_lda)
knn_best_lda.fit(X_train_lda, y_train_lda)
y_pred_knn_lda = knn_best_lda.predict(X_test_lda)
acc_knn_lda = accuracy_score(y_test_lda, y_pred_knn_lda)
print(f"KNN with LDA Accuracy (Best K={best_k_lda}): {acc_knn_lda:.4f}")


fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(knn_best_lda, X_test_lda, y_test_lda, ax=ax, cmap='Blues')
plt.title("KNN (LDA) - Confusion Matrix", fontsize=16)
plt.show()


fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(knn_best_lda, X_test_lda, y_test_lda, ax=ax)
plt.title("KNN (LDA) - ROC Curve", fontsize=16)
plt.show()


#-------------------------------------------------------------------




X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=101)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=101)

# ==================== 随机森林模型（不使用PCA） ==================== #
RFC_no_pca = RandomForestClassifier(random_state=101)
params_no_pca = {
    "n_estimators": [30, 64, 100, 128, 200],
    "max_features": [2, 3, 4],
    "max_depth": [4, 8, 10, 15]
}


grid_model_no_pca = GridSearchCV(RFC_no_pca, params_no_pca, return_train_score=True, verbose=2)
grid_model_no_pca.fit(X_train, y_train)


cv_results_no_pca = pd.DataFrame(grid_model_no_pca.cv_results_)

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


best_params_no_pca = grid_model_no_pca.best_params_
RFC_no_pca = RandomForestClassifier(random_state=101, **best_params_no_pca)
RFC_no_pca.fit(X_train, y_train)

y_pred_test_no_pca = RFC_no_pca.predict(X_test)
y_pred_train_no_pca = RFC_no_pca.predict(X_train)


acc_rfc_no_pca = accuracy_score(y_test, y_pred_test_no_pca)


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
#-------------------------------------------------------------------
# ==================== 随机森林模型（使用PCA） ==================== #
RFC_pca = RandomForestClassifier(random_state=101)
params_pca = {
    "n_estimators": [30, 64, 100, 128, 200],
    "max_features": [2, 3, 4],
    "max_depth": [4, 8, 10, 15]
}


grid_model_pca = GridSearchCV(RFC_pca, params_pca, return_train_score=True, verbose=2)
grid_model_pca.fit(X_train_pca, y_train_pca)

cv_results_pca = pd.DataFrame(grid_model_pca.cv_results_)

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


best_params_pca = grid_model_pca.best_params_
RFC_pca = RandomForestClassifier(random_state=101, **best_params_pca)
RFC_pca.fit(X_train_pca, y_train_pca)


y_pred_test_pca = RFC_pca.predict(X_test_pca)
y_pred_train_pca = RFC_pca.predict(X_train_pca)


acc_rfc_pca = accuracy_score(y_test_pca, y_pred_test_pca)


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


RFC_lda = RandomForestClassifier(random_state=101)
params_lda = {
    "n_estimators": [30, 64, 100, 128, 200],
    "max_features": [2, 3, 4],
    "max_depth": [4, 8, 10, 15]
}

grid_model_lda = GridSearchCV(RFC_lda, params_lda, return_train_score=True, verbose=2)
grid_model_lda.fit(X_train_lda, y_train_lda)

best_params_lda = grid_model_lda.best_params_
print(f"Best Params (LDA): {best_params_lda}")


RFC_lda = RandomForestClassifier(random_state=101, **best_params_lda)
RFC_lda.fit(X_train_lda, y_train_lda)
y_pred_test_lda = RFC_lda.predict(X_test_lda)


acc_rfc_lda = accuracy_score(y_test_lda, y_pred_test_lda)
print("Test Accuracy (LDA):", acc_rfc_lda)


fig, ax = plt.subplots(figsize=(10, 6))
ConfusionMatrixDisplay.from_estimator(RFC_lda, X_test_lda, y_test_lda, ax=ax, cmap='Blues')
plt.title("Random Forest (LDA) - Confusion Matrix", fontsize=16)
plt.show()


fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
RocCurveDisplay.from_estimator(RFC_lda, X_test_lda, y_test_lda, ax=ax)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.title("Random Forest (LDA) - ROC Curve", fontsize=16)
plt.show()



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


labels = ['KNN (No PCA)', 'KNN (PCA)', 'KNN (LDA)', 'RF (No PCA)', 'RF (PCA)', 'RF (LDA)']
accuracies = [acc_knn_no_pca, acc_knn_pca, acc_knn_lda, acc_rfc_no_pca, acc_rfc_pca, acc_rfc_lda]


plt.figure(figsize=(4.5, 3.5), dpi=300)
sns.barplot(x=labels, y=accuracies, palette="coolwarm")
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Comparison (With PCA/LDA)', fontsize=12)
plt.ylim(0.8, 1)
plt.xticks(rotation=15, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()
