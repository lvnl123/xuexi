import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

output_dir = "03特征工程 tupian"
os.makedirs(output_dir, exist_ok=True)

train_data_file = "./zhengqi_train.txt"
test_data_file = "./zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

train_data.describe()

plt.figure(figsize=(18, 10))
plt.boxplot(x=train_data.values, labels=train_data.columns)
plt.hlines([-7.5, 7.5], 0, 40, colors='r')
plt.savefig(os.path.join(output_dir, '01异常值分析'))
plt.show()

train_data = train_data[train_data['V9'] > -7.5]
train_data.describe()

test_data.describe()

from sklearn import preprocessing

features_columns = [col for col in train_data.columns if col not in ['target']]
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(train_data[features_columns])
train_data_scaler = min_max_scaler.transform(train_data[features_columns])
test_data_scaler = min_max_scaler.transform(test_data[features_columns])
train_data_scaler = pd.DataFrame(train_data_scaler)
train_data_scaler.columns = features_columns
test_data_scaler = pd.DataFrame(test_data_scaler)
test_data_scaler.columns = features_columns
train_data_scaler['target'] = train_data['target']

train_data_scaler.describe()

test_data_scaler.describe()

dist_cols = 6
dist_rows = len(test_data_scaler.columns)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
for i, col in enumerate(test_data_scaler.columns):
    ax = plt.subplot(dist_rows, dist_cols, i + 1)
    ax = sns.kdeplot(train_data_scaler[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data_scaler[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(["train", "test"])
plt.savefig(os.path.join(output_dir, '02总密度分布图'))
plt.show()

drop_col = 6
drop_row = 1
plt.figure(figsize=(5 * drop_col, 5 * drop_row))
for i, col in enumerate(["V5", "V9", "V11", "V17", "V22", "V28"]):
    ax = plt.subplot(drop_row, drop_col, i + 1)
    ax = sns.kdeplot(train_data_scaler[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data_scaler[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(["train", "test"])
plt.savefig(os.path.join(output_dir, '03单条密度分布图'))
plt.show()

plt.figure(figsize=(20, 16))
column = train_data_scaler.columns.tolist()
mcorr = train_data_scaler[column].corr(method="spearman")
mask = np.zeros_like(mcorr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
plt.savefig(os.path.join(output_dir, '04特征相关性热力图'))
plt.show()

mcorr = mcorr.abs()
numerical_corr = mcorr[mcorr['target'] > 0.1]['target']
print(numerical_corr.sort_values(ascending=False))
index0 = numerical_corr.sort_values(ascending=False).index
print(train_data_scaler[index0].corr('spearman'))

features_corr = numerical_corr.sort_values(ascending=False).reset_index()
features_corr.columns = ['features_and_target', 'corr']
features_corr_select = features_corr[features_corr['corr'] > 0.3]  # 筛选出大于相关性大于0.3的特征
print(features_corr_select)
select_features = [col for col in features_corr_select['features_and_target'] if col not in ['target']]
new_train_data_corr_select = train_data_scaler[select_features + ['target']]
new_test_data_corr_select = test_data_scaler[select_features]

from statsmodels.stats.outliers_influence import variance_inflation_factor  # 多重共线性方差膨胀因子

# 多重共线性
new_numerical = ['V0', 'V2', 'V3', 'V4', 'V5', 'V6', 'V10', 'V11',
                 'V13', 'V15', 'V16', 'V18', 'V19', 'V20', 'V22', 'V24', 'V30', 'V31', 'V37']
X = np.matrix(train_data_scaler[new_numerical])
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
VIF_list

from sklearn.decomposition import PCA  # 主成分分析法

# PCA方法降维
# 保持90%的信息
pca = PCA(n_components=0.9)
new_train_pca_90 = pca.fit_transform(train_data_scaler.iloc[:, 0:-1])
new_test_pca_90 = pca.transform(test_data_scaler)
new_train_pca_90 = pd.DataFrame(new_train_pca_90)
new_test_pca_90 = pd.DataFrame(new_test_pca_90)
new_train_pca_90['target'] = train_data_scaler['target']
new_train_pca_90.describe()

train_data_scaler.describe()

# PCA方法降维
# 保留16个主成分
pca = PCA(n_components=0.95)
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:, 0:-1])
new_test_pca_16 = pca.transform(test_data_scaler)
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pca_16 = pd.DataFrame(new_test_pca_16)
new_train_pca_16['target'] = train_data_scaler['target']
new_train_pca_16.describe()

from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.neighbors import KNeighborsRegressor  # K近邻回归
from sklearn.tree import DecisionTreeRegressor  # 决策树回归
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
from sklearn.svm import SVR  # 支持向量回归
import lightgbm as lgb  # lightGbm模型
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split  # 切分数据
from sklearn.metrics import mean_squared_error  # 评价指标

# 采用 pca 保留16维特征的数据
new_train_pca_16 = new_train_pca_16.fillna(0)
train = new_train_pca_16[new_test_pca_16.columns]
target = new_train_pca_16['target']
# 切分数据 训练数据80% 验证数据20%
train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)

clf = LinearRegression()
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("LinearRegression:   ", score)

train_score = []
test_score = []
# 给予不同的数据量，查看模型的学习效果
for i in range(10, len(train_data) + 1, 10):
    lin_reg = LinearRegression()
    lin_reg.fit(train_data[:i], train_target[:i])
    y_train_predict = lin_reg.predict(train_data[:i])
    train_score.append(mean_squared_error(train_target[:i], y_train_predict))
    y_test_predict = lin_reg.predict(test_data)
    test_score.append(mean_squared_error(test_target, y_test_predict))

plt.plot([i for i in range(1, len(train_score) + 1)], train_score, label='train')
plt.plot([i for i in range(1, len(test_score) + 1)], test_score, label='test')
plt.legend()
plt.savefig(os.path.join(output_dir, '05线性回归图'))
plt.show()


def plot_learning_curve(algo, X_train, X_test, y_train, y_test, filename):
    """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
    """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
    train_score = []
    test_score = []
    for i in range(10, len(X_train) + 1, 10):
        algo.fit(X_train[:i], y_train[:i])
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(train_score) + 1)],
             train_score, label="train")
    plt.plot([i for i in range(1, len(test_score) + 1)],
             test_score, label="test")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


# 第6张图
plot_learning_curve(LinearRegression(), train_data, test_data, train_target, test_target, '06线性回归2图')

# 第7张图
plot_learning_curve(KNeighborsRegressor(n_neighbors=5), train_data, test_data, train_target, test_target,
                              '07K近邻回归图')

# 第8张图
plot_learning_curve(DecisionTreeRegressor(), train_data, test_data, train_target, test_target,
                              '08决策树回归图')

# 后续模型评估部分保持不变
clf = DecisionTreeRegressor()
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("DecisionTreeRegressor:   ", score)

clf = RandomForestRegressor(n_estimators=200)  # 200棵树模型
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("RandomForestRegressor:   ", score)

from sklearn.ensemble import GradientBoostingRegressor

myGBR = GradientBoostingRegressor(
    alpha=0.9,
    criterion='friedman_mse',
    init=None,
    learning_rate=0.03,
    loss='huber',
    max_depth=14,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=10,
    min_samples_split=40,
    min_weight_fraction_leaf=0.0,
    n_estimators=300,
    random_state=10,
    subsample=0.8,
    verbose=0,
    warm_start=False
)
myGBR.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("GradientBoostingRegressor:   ", score)