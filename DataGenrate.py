'加载原始数据后，先使用IG和Correlation算法进行特征选择，然后使用孤独森林算法分别对每一类的数据进行异常值去除，最后分别保存成train_data.npy和train_label.npy文件'

print(__doc__)

import pandas as pd
import numpy as np

train_file = 'G:/Machine Learning/NIDS/UNSW_NB15_training-set.csv'
test_file = 'G:/Machine Learning/NIDS/UNSW_NB15_testing-set.csv'
raw_train = np.array(pd.read_csv(train_file, sep='\t', header=None))
raw_test = np.array(pd.read_csv(test_file, sep='\t', header=None))
print(raw_train.shape)
print(raw_test.shape)

# 经过IG和Correlation特征序号
# feature_index = [2, 3, 6, 7, 8, 9, 10, 11, 12, 19, 22, 24, 31, 32, 34, 35, 40]
feature_index = [1, 2, 5, 6, 7, 8, 9, 10, 11, 18, 21, 23, 30, 31, 33, 34, 39]
featured_raw_train = raw_train[:, feature_index]
feature_raw_test = raw_test[:, feature_index]
label_train = raw_train[:, 43]
label_test = raw_test[:, 43]
total_data = np.r_[featured_raw_train, feature_raw_test]

# 对数据进行归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
featured_normalized_total = min_max_scaler.fit_transform(total_data)
featured_normalized_train = featured_normalized_total[0: 175341, :]
featured_normalized_test = featured_normalized_total[175341:, :]

# 分别存储数据
normal_data = featured_normalized_train[np.where(raw_train[:, 42] == 0)[0], :]
test_normal_data = featured_normalized_test[np.where(raw_test[:, 42] == 0)[0], :]
df_normal_data = pd.DataFrame(normal_data)
normal_data = np.array(df_normal_data.sample(frac=0.3, replace=False))
label_normal = np.repeat(0, len(normal_data))

fuzzer_data = featured_normalized_train[np.where(raw_train[:, 42] == 1)[0], :]
test_fuzzer_data = featured_normalized_test[np.where(raw_test[:, 42] == 1)[0], :]
df_fuzzer_data = pd.DataFrame(fuzzer_data)
fuzzer_data = np.array(df_fuzzer_data.sample(frac=0.3, replace=False))
label_fuzzer = np.repeat(1, len(fuzzer_data))

exploit_data = featured_normalized_train[np.where(raw_train[:, 42] == 2)[0], :]
test_exploit_data = featured_normalized_test[np.where(raw_test[:, 42] == 2)[0], :]
df_expoit_data = pd.DataFrame(exploit_data)
exploit_data = np.array(df_expoit_data.sample(frac=0.3, replace=False))
label_exploit = np.repeat(2, len(exploit_data))

reconna_data = featured_normalized_train[np.where(raw_train[:, 42] == 3)[0], :]
test_reconna_data = featured_normalized_test[np.where(raw_test[:, 42] == 3)[0], :]
df_reconna_data = pd.DataFrame(reconna_data)
reconna_data = np.array(df_reconna_data.sample(frac=0.3, replace=False))
label_reconna = np.repeat(3, len(reconna_data))

shellcode_data = featured_normalized_train[np.where(raw_train[:, 42] == 4)[0], :]
test_shellcode_data = featured_normalized_test[np.where(raw_test[:, 42] == 4)[0], :]
df_shellcode_data = pd.DataFrame(shellcode_data)
shellcode_data = np.array(df_shellcode_data.sample(frac=0.3, replace=False))
label_shellcode = np.repeat(4, len(shellcode_data))

ddos_data = featured_normalized_train[np.where(raw_train[:, 42] == 5)[0], :]
test_ddos_data = featured_normalized_test[np.where(raw_test[:, 42] == 5)[0], :]
df_ddos_data = pd.DataFrame(ddos_data)
ddos_data = np.array(df_ddos_data.sample(frac=0.3, replace=False))
label_ddos = np.repeat(5, len(ddos_data))

analysis_data = featured_normalized_train[np.where(raw_train[:, 42] == 6)[0], :]
test_analysis_data = featured_normalized_test[np.where(raw_test[:, 42] == 6)[0], :]
df_analysis_data = pd.DataFrame(analysis_data)
analysis_data = np.array(df_analysis_data.sample(frac=0.3, replace=False))
label_analysis = np.repeat(6, len(analysis_data))

backdoor_data = featured_normalized_train[np.where(raw_train[:, 42] == 7)[0], :]
test_backdoor_data = featured_normalized_test[np.where(raw_test[:, 42] == 7)[0], :]
df_backdoor_data = pd.DataFrame(backdoor_data)
backdoor_data = np.array(df_backdoor_data.sample(frac=0.3, replace=False))
label_backdoor = np.repeat(7, len(backdoor_data))

worms_data = featured_normalized_train[np.where(raw_train[:, 42] == 8)[0], :]
test_worms_data = featured_normalized_test[np.where(raw_test[:, 42] == 8)[0], :]
df_worms_data = pd.DataFrame(worms_data)
worms_data = np.array(df_worms_data.sample(frac=0.3, replace=False))
label_worms = np.repeat(8, len(worms_data))

generic_data = featured_normalized_train[np.where(raw_train[:, 42] == 9)[0], :]
test_generic_data = featured_normalized_test[np.where(raw_test[:, 42] == 9)[0], :]
df_generic_data = pd.DataFrame(generic_data)
generic_data = np.array(df_generic_data.sample(frac=0.3, replace=False))
label_generic = np.repeat(9, len(generic_data))

train_fixed_data = np.r_[normal_data, fuzzer_data, exploit_data, reconna_data, shellcode_data,
              ddos_data, analysis_data, backdoor_data, worms_data, generic_data]

train_fixed_label = np.r_[label_normal, label_fuzzer, label_exploit, label_reconna, label_shellcode,
                label_ddos, label_analysis, label_backdoor, label_worms, label_generic]


test_fixed_data = np.r_[test_normal_data, test_fuzzer_data, test_exploit_data, test_reconna_data, test_shellcode_data,
                test_ddos_data, test_analysis_data, test_backdoor_data, test_worms_data, test_generic_data]

test_fixed_label = np.r_[test_label_normal, test_label_fuzzer, test_label_exploit, test_label_reconna, test_label_shellode,
    test_label_ddos, test_label_analysis, test_label_backdoor, test_label_worms, test_label_generic]


print('正在保存数据....')
np.save('train_data_17W_0.1.npy', train_fixed_data)
np.save('train_label_17W_0.1.npy', train_fixed_label)
np.save('test_label_8W.npy', test_fixed_label)
np.save('test_data_8W.npy', test_fixed_data)
print('保存数据完毕...')

