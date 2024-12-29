import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    #读取ARFF文件，将数据加载为元组，第一个元素是数据，第二个元素是元数据
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)#将读取的数据转换为 pandas DataFrame，便于后续操作
    # 如果数据中含有byte类型的字符串，则需要解码
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.decode('utf-8')
    return df


# 存入数据
train_data = load_arff_data('D:\study\code\python\机器学习大作业\ECG5000\ECG5000_TRAIN.arff')
test_data = load_arff_data('D:\study\code\python\机器学习大作业\ECG5000\ECG5000_TEST.arff')

# 分割训练集和测试集数据
X_train = train_data.iloc[:, :-1]  # 特征列
y_train = train_data.iloc[:, -1]  # 标签列
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]


# SVM支持向量机，使用线性核函数
def svm_classifier(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

#决策树
def decision_tree_classifier(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

#随机森林，使用 100 棵树
def random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# 训练模型并评估
def evaluate_model(model, X_test, y_test, model_name="Model"):
    #预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')  # 加权召回率
    precision = precision_score(y_test, y_pred, average='weighted')  # 加权精确率
    report = classification_report(y_test, y_pred)

    print(f"{model_name} 分类准确率: {accuracy:.4f}")
    print(f"{model_name} 召回率: {recall:.4f}")
    print(f"{model_name} 精确率: {precision:.4f}")
    print(f"{model_name} 分类报告:\n{report}")

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)#计算混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"{model_name} 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

    return accuracy, y_pred


# 评估并比较所有模型
def compare_models(X_train, y_train, X_test, y_test):
    models = {
        'SVM': svm_classifier(X_train, y_train),
        'Decision Tree': decision_tree_classifier(X_train, y_train),
        'Random Forest': random_forest_classifier(X_train, y_train),
    }

    results = {}

    # 评估每个模型
    for model_name, model in models.items():
        accuracy, _ = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = accuracy

    # 输出最佳模型
    best_model = max(results, key=results.get)
    print(f"\n最优分类模型是: {best_model}，准确率为: {results[best_model]:.4f}")
    return best_model, results


# 主函数
def main():
    # 比较所有模型并输出最佳分类器
    best_model, results = compare_models(X_train, y_train, X_test, y_test)
    print("\n所有模型的准确率比较：")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
