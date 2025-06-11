import pandas as pd

train_data_file = "./zhengqi_train.txt"
test_data_file = "./zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

epsilon = 1e-5
# 组交叉特征，可以自行定义，如增加： x*x/y, log(x)/y 等等
func_dict = {
    'add': lambda x, y: x + y,
    'mins': lambda x, y: x - y,
    'div': lambda x, y: x / (y + epsilon),
    'multi': lambda x, y: x * y
}

def auto_features_make(train_data, test_data, func_dict, col_list):
    train_data, test_data = train_data.copy(), test_data.copy()

    # 用字典暂存新特征，避免多次直接赋值导致碎片化警告
    train_new_features = {}
    test_new_features = {}

    for col_i in col_list:
        for col_j in col_list:
            for func_name, func in func_dict.items():
                col_func_features = '-'.join([col_i, func_name, col_j])
                train_new_features[col_func_features] = func(train_data[col_i], train_data[col_j])
                test_new_features[col_func_features] = func(test_data[col_i], test_data[col_j])

    # 一次性合并
    train_new_df = pd.DataFrame(train_new_features)
    test_new_df = pd.DataFrame(test_new_features)
    train_data = pd.concat([train_data, train_new_df], axis=1)
    test_data = pd.concat([test_data, test_new_df], axis=1)

    return train_data, test_data

# 确保传入的列是特征列（不含target）
col_list = [col for col in test_data.columns]

train_data2, test_data2 = auto_features_make(train_data, test_data, func_dict, col_list=col_list)

from sklearn.decomposition import PCA  # 主成分分析法

# 训练集特征列（排除target）
feature_cols = [col for col in train_data2.columns if col != 'target']

pca = PCA(n_components=500)
train_data2_pca = pca.fit_transform(train_data2[feature_cols])
test_data2_pca = pca.transform(test_data2[feature_cols])

train_data2_pca = pd.DataFrame(train_data2_pca)
test_data2_pca = pd.DataFrame(test_data2_pca)
train_data2_pca['target'] = train_data2['target']

X_train2 = train_data2_pca.drop(columns=['target'])
y_train = train_data2_pca['target'].values

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
from lightgbm import early_stopping, log_evaluation

# 5折交叉验证
Folds = 5
kf = KFold(n_splits=Folds, random_state=2019, shuffle=True)

MSE_DICT = {
    'train_mse': [],
    'test_mse': []
}

for i, (train_index, test_index) in enumerate(kf.split(X_train2)):
    lgb_reg = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2019,
        objective='regression',
    )

    # 切分训练集和测试集，保持DataFrame和列名
    X_train_KFold = X_train2.iloc[train_index]
    X_test_KFold = X_train2.iloc[test_index]
    y_train_KFold = y_train[train_index]
    y_test_KFold = y_train[test_index]

    lgb_reg.fit(
        X=X_train_KFold, y=y_train_KFold,
        eval_set=[(X_train_KFold, y_train_KFold), (X_test_KFold, y_test_KFold)],
        eval_names=['Train', 'Test'],
        eval_metric='l2',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=50)
        ]
    )

    y_train_KFold_predict = lgb_reg.predict(X_train_KFold, num_iteration=lgb_reg.best_iteration_)
    y_test_KFold_predict = lgb_reg.predict(X_test_KFold, num_iteration=lgb_reg.best_iteration_)

    print(f'第{i}折 训练和预测 训练MSE 预测MSE')
    train_mse = mean_squared_error(y_train_KFold_predict, y_train_KFold)
    print('------\n训练MSE\n', train_mse, '\n------')
    test_mse = mean_squared_error(y_test_KFold_predict, y_test_KFold)
    print('------\n预测MSE\n', test_mse, '\n------\n')

    MSE_DICT['train_mse'].append(train_mse)
    MSE_DICT['test_mse'].append(test_mse)

print('------\n训练MSE\n', MSE_DICT['train_mse'], '\n平均:', np.mean(MSE_DICT['train_mse']), '\n------')
print('------\n预测MSE\n', MSE_DICT['test_mse'], '\n平均:', np.mean(MSE_DICT['test_mse']), '\n------')
