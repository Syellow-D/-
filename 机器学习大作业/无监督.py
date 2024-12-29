import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
#忽略一些提醒
import warnings
warnings.filterwarnings("ignore")
#设置图的字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取ARFF文件，将数据加载为元组，第一个元素是数据，第二个元素是元数据
data, meta = arff.loadarff('D:\study\code\python\机器学习大作业\ECG5000\ECG5000_TEST.arff')  # 替换为你的路径

# 转换为pandas DataFrame
df = pd.DataFrame(data)

# 最后一列为类别，其他列为特征
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].apply(lambda x: x.decode('utf-8')).values  # 标签数据，解码为字符串格式

# 使用LabelEncoder将标签转换为数字格式，否则后面会报错
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) + 1  # 标签转换为1到5范围内

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化数据

#KMeans
n_clusters = 5  # 设置类别数量为5
kmeans = KMeans(n_clusters=n_clusters, random_state=9)

# 拟合KMeans模型
kmeans.fit(X_scaled)

# 预测类别
y_pred = kmeans.predict(X_scaled)
y_pred_mapped = y_pred + 1  # 将预测类别范围从 [0, 4] 映射到 [1, 5]

# 评估模型效果
print("\n分类报告:")
print(classification_report(y_encoded, y_pred_mapped))

# 计算准确率
accuracy = accuracy_score(y_encoded, y_pred_mapped)
print("分类准确率: {:.4f}".format(accuracy))

# 计算召回率
recall = recall_score(y_encoded, y_pred_mapped, average='weighted')
print("召回率: {:.4f}".format(recall))

# 计算精确率
precision = precision_score(y_encoded, y_pred_mapped, average='weighted')
print("精确率: {:.4f}".format(precision))

# 可视化
# 生成混淆矩阵
conf_matrix = confusion_matrix(y_encoded, y_pred_mapped)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, n_clusters+1), yticklabels=np.arange(1, n_clusters+1))
plt.title('混淆矩阵 (Confusion Matrix)', fontsize=14)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.show()

