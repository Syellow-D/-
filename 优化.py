import pandas as pd
import numpy as np
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
#忽略一些提醒
import warnings
warnings.filterwarnings("ignore")
#设置图的字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取ARFF文件
def load_arff_data(file_path):
    #第一个元素是数据，第二个元素是元数据
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # 如果数据中含有byte类型的字符串，则需要解码
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.decode('utf-8')
    return df

# 载入数据
train_data = load_arff_data('D:\\study\\code\\python\\机器学习大作业\\ECG5000\\ECG5000_TRAIN.arff')
test_data = load_arff_data('D:\\study\\code\\python\\机器学习大作业\\ECG5000\\ECG5000_TEST.arff')

# 分割数据
X_train = train_data.iloc[:, :-1]  # 特征列
y_train = train_data.iloc[:, -1]  # 标签列
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA降维，将数据降到50个主成分
def apply_pca(X_train, X_test, n_components=50):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled)

# 使用GridSearchCV调优SVM参数
def optimize_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# 定义模型函数
def svm_classifier(X_train, y_train):
    best_model = optimize_svm(X_train, y_train)
    return best_model

# 评估模型
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    print(f"{model_name} 分类准确率: {accuracy:.4f}")
    print(f"{model_name} 召回率: {recall:.4f}")
    print(f"{model_name} 精确率: {precision:.4f}")
    print(f"{model_name} 分类报告:\n{report}")

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"{model_name} 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

    return accuracy, y_pred

# 主函数
def main():
    model = svm_classifier(X_train_pca, y_train)
    evaluate_model(model, X_test_pca, y_test, "SVM")

if __name__ == "__main__":
    main()
