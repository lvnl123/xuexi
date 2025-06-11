import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings("ignore")

# 创建保存图片的目录
output_dir = "./02 tupian"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
train_data_file = "./zhengqi_train.txt"
test_data_file = "./zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

# 查看基本信息
print("Train Data Info:")
train_data.info()
print("\nTest Data Info:")
test_data.info()

print("\nTrain Data Describe:")
print(train_data.describe())
print("\nTest Data Describe:")
print(test_data.describe())

print("\nFirst 5 rows of Train Data:")
print(train_data.head())
print("\nFirst 5 rows of Test Data:")
print(test_data.head())

# 图1: 单个特征箱线图
plt.figure(figsize=(4, 6))
sns.boxplot(train_data['V0'], orient="v", width=0.5)
plt.title('Boxplot of V0')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_1_boxplot_v0.png'))
plt.show()

# 图2: 所有特征箱线图
column = train_data.columns.tolist()[:39]
fig = plt.figure(figsize=(20, 40))
for i in range(38):
    plt.subplot(13, 3, i + 1)
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)
    plt.ylabel(column[i], fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_2_all_boxplots.png'))
plt.show()

# 图3: 分布图与概率图
plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 2, 1)
sns.distplot(train_data['V0'], fit=stats.norm)
ax = plt.subplot(1, 2, 2)
res = stats.probplot(train_data['V0'], plot=plt)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_3_dist_probplot_v0.png'))
plt.show()

# 图4: 所有特征分布图与概率图
train_cols = 6
train_rows = len(train_data.columns)
plt.figure(figsize=(4 * train_cols, 4 * train_rows))
i = 0
for col in train_data.columns:
    i += 1
    ax = plt.subplot(train_rows, train_cols, i)
    sns.distplot(train_data[col], fit=stats.norm)
    i += 1
    ax = plt.subplot(train_rows, train_cols, i)
    res = stats.probplot(train_data[col], plot=plt)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_4_all_dist_probplots.png'))
plt.show()

# 图5: KDE 对比图（V0）
ax = sns.kdeplot(train_data['V0'], color="Red", shade=True)
ax = sns.kdeplot(test_data['V0'], color="Blue", shade=True)
ax.set_xlabel('V0')
ax.set_ylabel("Frequency")
ax.legend(["train", "test"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_5_kdeplot_v0_compare.png'))
plt.show()

# 图6: 所有列的 KDE 对比图
dist_cols = 6
dist_rows = len(test_data.columns)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
i = 1
for col in test_data.columns:
    ax = plt.subplot(dist_rows, dist_cols, i)
    ax = sns.kdeplot(train_data[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(["train", "test"])
    i += 1
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_6_all_kdeplots.png'))
plt.show()

# 图7: 指定列 KDE 对比图
drop_columns = ["V5", "V9", "V11", "V17", "V22", "V28"]
drop_col_num = len(drop_columns)
plt.figure(figsize=(5 * drop_col_num, 5))
i = 1
for col in drop_columns:
    ax = plt.subplot(1, drop_col_num, i)
    ax = sns.kdeplot(train_data[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(["train", "test"])
    i += 1
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_7_selected_kdeplots.png'))
plt.show()

# 图8: 回归图和分布图
fcols = 2
frows = 1
plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
sns.regplot(x='V0', y='target', data=train_data, ax=ax,
            scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
            line_kws={'color': 'k'})
plt.xlabel('V0')
plt.ylabel('target')
ax = plt.subplot(1, 2, 2)
sns.distplot(train_data['V0'].dropna())
plt.xlabel('V0')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_8_regression_and_dist_v0.png'))
plt.show()

# 图9: 所有特征对 target 的回归图
fcols = 6
frows = len(test_data.columns)
plt.figure(figsize=(5 * fcols, 4 * frows))
i = 0
for col in test_data.columns:
    i += 1
    ax = plt.subplot(frows, fcols, i)
    sns.regplot(x=col, y='target', data=train_data, ax=ax,
                scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})
    plt.xlabel(col)
    plt.ylabel('target')
    i += 1
    ax = plt.subplot(frows, fcols, i)
    sns.distplot(train_data[col].dropna())
    plt.xlabel(col)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_9_all_regression_plots.png'))
plt.show()

# 图10: 相关性热力图
data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'], axis=1)
train_corr = data_train1.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(train_corr, vmax=.8, square=True, annot=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_10_correlation_heatmap.png'))
plt.show()

# 图11: Spearman 热力图
colnm = data_train1.columns.tolist()
mcorr = data_train1[colnm].corr(method="spearman")
mask = np.zeros_like(mcorr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(20, 16))
sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
plt.title('Spearman Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_11_spearman_heatmap.png'))
plt.show()

# 图12: Top K 特征热力图
k = 10
cols = train_corr.nlargest(k, 'target')['target'].index
cm = np.corrcoef(train_data[cols].values.T)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, cbar=True, annot=True, square=True)
plt.title('Top K Correlated Features Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_12_topk_correlation_heatmap.png'))
plt.show()

# 图13: 高相关特征热力图
threshold = 0.5
corrmat = train_data.corr()
top_corr_features = corrmat.index[abs(corrmat["target"]) > threshold]
plt.figure(figsize=(10, 10))
sns.heatmap(train_data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.title('High Correlation Features Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_13_high_corr_features_heatmap.png'))
plt.show()

# 数据归一化函数
def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())

# 提取数值型特征列（去掉 'target'）
cols_numeric = list(train_data.columns)
if 'target' in cols_numeric:
    cols_numeric.remove('target')  # 如果有 target 列，则去掉

# 对训练集进行归一化处理
train_data_process = train_data[cols_numeric].copy()
train_data_process = train_data_process[cols_numeric].apply(scale_minmax, axis=0)

# 添加 target 列用于后续绘图分析
train_data_process['target'] = train_data['target']

# 拆分特征列表用于可视化（左右两部分）
cols_numeric_left = cols_numeric[:13]
cols_numeric_right = cols_numeric[13:]

# 设置绘图参数
fcols = 6
frows = len(cols_numeric_left)

# 图14: Box-Cox 变换效果展示图（左边部分）
plt.figure(figsize=(4*fcols, 4*frows))
i = 0
for var in cols_numeric_left:
    dat = train_data_process[[var, 'target']].dropna()
    if len(dat[var]) < 2:
        continue  # 跳过数据太少的列

    i += 1
    plt.subplot(frows, fcols, i)
    sns.distplot(dat[var], fit=stats.norm)
    plt.title(var + ' Original')
    plt.xlabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    stats.probplot(dat[var], plot=plt)
    plt.title('skew={:.4f}'.format(stats.skew(dat[var])))
    plt.xlabel('')
    plt.ylabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(dat[var], dat['target'], '.', alpha=0.5)
    plt.title('corr={:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))

    i += 1
    trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
    trans_var = scale_minmax(trans_var)

    plt.subplot(frows, fcols, i)
    sns.distplot(trans_var, fit=stats.norm)
    plt.title(var + ' Transformed')
    plt.xlabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    stats.probplot(trans_var, plot=plt)
    plt.title('skew={:.4f}'.format(stats.skew(trans_var)))
    plt.xlabel('')
    plt.ylabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(trans_var, dat['target'], '.', alpha=0.5)
    plt.title('corr={:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_14_boxcox_transforms_left.png'))
plt.show()

# 图15: Box-Cox 变换效果展示图（右边部分）
frows = len(cols_numeric_right)
plt.figure(figsize=(4*fcols, 4*frows))
i = 0
for var in cols_numeric_right:
    dat = train_data_process[[var, 'target']].dropna()
    if len(dat[var]) < 2:
        continue  # 跳过数据太少的列

    i += 1
    plt.subplot(frows, fcols, i)
    sns.distplot(dat[var], fit=stats.norm)
    plt.title(var + ' Original')
    plt.xlabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    stats.probplot(dat[var], plot=plt)
    plt.title('skew={:.4f}'.format(stats.skew(dat[var])))
    plt.xlabel('')
    plt.ylabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(dat[var], dat['target'], '.', alpha=0.5)
    plt.title('corr={:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))

    i += 1
    trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
    trans_var = scale_minmax(trans_var)

    plt.subplot(frows, fcols, i)
    sns.distplot(trans_var, fit=stats.norm)
    plt.title(var + ' Transformed')
    plt.xlabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    stats.probplot(trans_var, plot=plt)
    plt.title('skew={:.4f}'.format(stats.skew(trans_var)))
    plt.xlabel('')
    plt.ylabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(trans_var, dat['target'], '.', alpha=0.5)
    plt.title('corr={:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_15_boxcox_transforms_right.png'))
plt.show()