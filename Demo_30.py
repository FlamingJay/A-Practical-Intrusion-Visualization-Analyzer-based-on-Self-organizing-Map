'二分类，30%的训练集，想要看一下分布'

import pandas as pd
import numpy as np
import time
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import sys
import pickle

# 加载预处理文件
train_data = np.load('sample2_train_data.npy')
train_label = np.load('sample2_train_label.npy').astype(int)
if train_data.shape[0] == train_label.shape[0]:
    print('行数相同....')

for i in range(len(train_label)):
    if train_label[i] != 0:
        train_label[i] = 1

# 创建新模型
from minisom import MiniSom
som = MiniSom(30, 30, 17, sigma=1.0, learning_rate=1.0)
print('SOM Training....')
som.pca_weights_init(train_data)

# 加载之前的模型
# print('加载之前的模型...')
# with open('SOM_sample_2class_200W.p', 'rb') as infile:
#     som = pickle.load(infile)

print('加载完毕，进行训练')
tic = time.process_time()
som.train_batch(train_data, 2000000)
toc = time.process_time()
print('模型训练完毕...,用时：%5.3f ms' % (1000 * (toc - tic)))

print('保存当前模型...')
import pickle
with open('SOM_sample2_30_30_200W.p', 'wb') as outfile:
    pickle.dump(som, outfile)
print('保存完毕.....')

# 画训练集的饼状图
tic = time.process_time()
labels_map = som.labels_map(train_data, train_label)
label_names = np.unique(train_label)  # 按照顺序输出，所以是[0]：0和[1]:1
#
plt.figure(figsize=(10, 10))
the_grid = GridSpec(30, 30)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[29-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
plt.legend(patches, label_names, loc='best')
plt.savefig('SOM_sample2_30_30_200W.png')
toc = time.process_time()
print('训练过程完毕...,用时：%5.3f ms' % (1000 * (toc - tic)))

#
# # # 画出误差函数曲线
# plt.figure(figsize=(10, 10))
# max_iter = 40000
# q_error_pca_init = []
# iter_x = []
# for i in range(max_iter):
#     percent = 100*(i+1)/max_iter
#     rand_i = np.random.randint(len(train_data))
#     som.update(train_data[rand_i], som.winner(train_data[rand_i]), i, max_iter)
#     if (i+1) % 100 == 0:
#         error = som.quantization_error(train_data)
#         q_error_pca_init.append(error)
#         iter_x.append(i)
#         sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')
# plt.plot(iter_x, q_error_pca_init)
# plt.ylabel('Quantization Error')
# plt.xlabel('Iteration Index')
# plt.savefig('Quantization Error in training process.png')

# # 进行检测
test_data = np.load('Isolation_test_data_17W0.1.npy')
test_label = np.load('Isolation_test_label_17W0.1.npy').astype(int)
for i in range(len(test_label)):
    if test_label[i] != 0:
        test_label[i] = 1

count = 0  # 计数与标签一致的个数
new = 0
tn = 0
tp = 0
fn = 0
fp = 0
predict_label = np.zeros(len(test_label),)
pos = labels_map
for cnt, xx in enumerate(test_data):
    print(cnt)
    w = som.winner(xx)  # 获得数据xx对应的神经元坐标
    if w in pos:
        label_fracs = [labels_map[w][l] for l in label_names]  # 分别获得各个标签在该位置上对应的数量
        label_1 = label_fracs[0] / sum(label_fracs)
        label_2 = label_fracs[1] / sum(label_fracs)
        predict_label[cnt] = np.where(np.max([label_1, label_2]) == [label_1, label_2])[0][0]  # 获得预测的标签
        if predict_label[cnt] == test_label[cnt]:
            count = count + 1

        if test_label[cnt] == 0:  # 如果是正常的
            if predict_label[cnt] == test_label[cnt]:
                tn = tn + 1
            else:
                fn = fn + 1
        else:  # 如果不是
            if predict_label[cnt] == test_label[cnt]:
                fp = fp + 1
            else:
                tp = tp + 1
    else:
        new = new + 1
#
np.save('predict_label_sample2_30_30_200W', predict_label)  # 保存预测的标签矩阵
T = [tp, tn, fp, fn]
np.save('T_sample2_30_30_200W', T)  # 保存TN什么的
print('count:%d' % count)
print('accuracy:%f' % (count / len(test_label)))
plt.show()
























#
# from minisom import MiniSom
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
# data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3), skip_header=True)
# data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
# som = MiniSom(7, 7, 4, sigma=3, learning_rate=0.5, neighborhood_function='triangle', random_seed=10)
# som.pca_weights_init(data)
# print("Training...")
# som.train_random(data, 4000)  # random training
# print("\n...ready!")
# plt.figure(figsize=(7, 7))
# plt.pcolor(som.distance_map().T, cmap='bone_r')
# target = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
# t = np.zeros(len(target), dtype=int)
# t[target == 'setosa'] = 0
# t[target == 'versicolor'] = 1
# t[target == 'virginica'] = 2
# markers = ['o', 's', 'D']
# colors = ['C0', 'C1', 'C2']
# for cnt, xx in enumerate(data):
#     w = som.winner(xx)  # getting the winner
#     plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
#              markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
# plt.axis([0, 7, 0, 7])
# # plt.savefig('som_iris.png')
# # plt.show()
#
# label = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str, skip_header=True)
# label_names = np.unique(label)
# labels_map = som.labels_map(data, label)
# x = labels_map[(2, 6)]
#
# plt.figure(figsize=(7, 7))
# the_grid = GridSpec(7, 7)
# for position in labels_map.keys():
#     label_fracs = [labels_map[position][l] for l in label_names] # 分别获得各个标签在该位置上对应的数量
#     label_1 = label_fracs[0] / sum(label_fracs)
#     label_2 = label_fracs[1] / sum(label_fracs)
#     label_3 = label_fracs[2] / sum(label_fracs)
#     index = np.where(np.max([label_1, label_2, label_3]))
#
#     plt.subplot(the_grid[6-position[1], position[0]], aspect=1)
#     patches, texts = plt.pie(label_fracs)
# plt.legend(patches, label_names, bbox_to_anchor=(0, 1), ncol=3)
# plt.savefig('som_iris_pies.png')
# plt.show()
#
# max_iter = 4000
# q_error_pca_init = []
# iter_x = []
# for i in range(max_iter):
#     percent = 100*(i+1)/max_iter
#     rand_i = np.random.randint(len(data))
#     som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
#     if (i+1) % 100 == 0:
#         error = som.quantization_error(data)
#         q_error_pca_init.append(error)
#         iter_x.append(i)
#         sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')
#
# plt.plot(iter_x, q_error_pca_init)
# plt.ylabel('quantization error')
# plt.xlabel('iteration index')
# plt.show()