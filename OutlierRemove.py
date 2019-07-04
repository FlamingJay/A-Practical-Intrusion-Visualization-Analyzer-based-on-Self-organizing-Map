import numpy as np

train_data = np.load('test_data_8W.npy')
train_label = np.load('test_label_8W.npy')
test_data = np.load('train_data_17W_0.1.npy')
test_label = np.load('train_label_17W_0.1.npy')


print('Start to Remove Outliers for each calss...')
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=300, max_samples=256, contamination=0.1)
# 去除对应下标的数据，并生成新数据
# Normal类型
normal_data = train_data[np.where(train_label == 0)[0], :]
test_normal_data = test_data[np.where(test_label == 0)[0], :]

clf.fit(normal_data)
y_pred_train = clf.predict(normal_data)
normal_fixed_data = np.delete(normal_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_normal = np.repeat(0, len(normal_fixed_data))

clf.fit(test_normal_data)
y_pred_train = clf.predict(test_normal_data)
test_normal_fixed_data = np.delete(test_normal_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_normal = np.repeat(0, len(test_normal_fixed_data))

# Fuzzer类型
fuzzer_data = train_data[np.where(train_label == 1)[0], :]
test_fuzzer_data = test_data[np.where(test_label == 1)[0], :]

clf.fit(fuzzer_data)
y_pred_train = clf.predict(fuzzer_data)
fuzzer_fixed_data = np.delete(fuzzer_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_fuzzer = np.repeat(1, len(fuzzer_fixed_data))

clf.fit(test_fuzzer_data)
y_pred_train = clf.predict(test_fuzzer_data)
test_fuzzer_fixed_data = np.delete(test_fuzzer_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_fuzzer = np.repeat(1, len(test_fuzzer_fixed_data))

# Exploit类型
exploit_data = train_data[np.where(train_label == 2)[0], :]
test_exploit_data = test_data[np.where(test_label == 2)[0], :]

clf.fit(exploit_data)
y_pred_train = clf.predict(exploit_data)
exploit_fixed_data = np.delete(exploit_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_exploit = np.repeat(2, len(exploit_fixed_data))

clf.fit(test_exploit_data)
y_pred_train = clf.predict(test_exploit_data)
test_exploit_fixed_data = np.delete(test_exploit_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_exploit = np.repeat(2, len(test_exploit_fixed_data))

# Reconna类型
reconna_data = train_data[np.where(train_label == 3)[0], :]
test_reconna_data = test_data[np.where(test_label == 3)[0], :]

clf.fit(reconna_data)
y_pred_train = clf.predict(reconna_data)
reconna_fixed_data = np.delete(reconna_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_reconna = np.repeat(3, len(reconna_fixed_data))

clf.fit(test_reconna_data)
y_pred_train = clf.predict(test_reconna_data)
test_reconna_fixed_data = np.delete(test_reconna_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_reconna = np.repeat(3, len(test_reconna_fixed_data))

# Shellcode类型
shellcode_data = train_data[np.where(train_label == 4)[0], :]
test_shellcode_data = test_data[np.where(test_label == 4)[0], :]

clf.fit(shellcode_data)
y_pred_train = clf.predict(shellcode_data)
shellcode_fixed_data = np.delete(shellcode_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_shellode = np.repeat(4, len(shellcode_fixed_data))

clf.fit(test_shellcode_data)
y_pred_train = clf.predict(test_shellcode_data)
test_shellcode_fixed_data = np.delete(test_shellcode_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_shellode = np.repeat(4, len(test_shellcode_fixed_data))

# Dos类型
ddos_data = train_data[np.where(train_label == 5)[0], :]
test_ddos_data = test_data[np.where(test_label == 5)[0], :]

clf.fit(ddos_data)
y_pred_train = clf.predict(ddos_data)
ddos_fixed_data = np.delete(ddos_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_ddos = np.repeat(5, len(ddos_fixed_data))

clf.fit(test_ddos_data)
y_pred_train = clf.predict(test_ddos_data)
test_ddos_fixed_data = np.delete(test_ddos_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_ddos = np.repeat(5, len(test_ddos_fixed_data))

# Analysis类型
analysis_data = train_data[np.where(train_label == 6)[0], :]
test_analysis_data = test_data[np.where(test_label == 6)[0], :]

clf.fit(analysis_data)
y_pred_train = clf.predict(analysis_data)
analysis_fixed_data = np.delete(analysis_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_analysis = np.repeat(6, len(analysis_fixed_data))

clf.fit(test_analysis_data)
y_pred_train = clf.predict(test_analysis_data)
test_analysis_fixed_data = np.delete(test_analysis_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_analysis = np.repeat(6, len(test_analysis_fixed_data))

# Backdoor类型
backdoor_data = train_data[np.where(train_label == 7)[0], :]
test_backdoor_data = test_data[np.where(test_label == 7)[0], :]

clf.fit(backdoor_data)
y_pred_train = clf.predict(backdoor_data)
backdoor_fixed_data = np.delete(backdoor_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_backdoor = np.repeat(7, len(backdoor_fixed_data))

clf.fit(test_backdoor_data)
y_pred_train = clf.predict(test_backdoor_data)
test_backdoor_fixed_data = np.delete(test_backdoor_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_backdoor = np.repeat(7, len(test_backdoor_fixed_data))

# Worms类型
worms_data = train_data[np.where(train_label == 8)[0], :]
test_worms_data = test_data[np.where(test_label == 8)[0], :]

clf.fit(worms_data)
y_pred_train = clf.predict(worms_data)
worms_fixed_data = np.delete(worms_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_worms = np.repeat(8, len(worms_fixed_data))

clf.fit(test_worms_data)
y_pred_train = clf.predict(test_worms_data)
test_worms_fixed_data = np.delete(test_worms_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_worms = np.repeat(8, len(test_worms_fixed_data))

# Generic类型
generic_data = train_data[np.where(train_label == 9)[0], :]
test_generic_data = test_data[np.where(test_label == 9)[0], :]

clf.fit(generic_data)
y_pred_train = clf.predict(generic_data)
generic_fixed_data = np.delete(generic_data, np.where(y_pred_train == -1)[0], axis=0)
label_fixed_generic = np.repeat(9, len(generic_fixed_data))

clf.fit(test_generic_data)
y_pred_train = clf.predict(test_generic_data)
test_generic_fixed_data = np.delete(test_generic_data, np.where(y_pred_train == -1)[0], axis=0)
test_label_fixed_generic = np.repeat(9, len(test_generic_fixed_data))


train_fixed_data = np.r_[normal_fixed_data, fuzzer_fixed_data, exploit_fixed_data, reconna_fixed_data, shellcode_fixed_data,
              ddos_fixed_data, analysis_fixed_data, backdoor_fixed_data, worms_fixed_data, generic_fixed_data]

train_fixed_label = np.r_[label_fixed_normal, label_fixed_fuzzer, label_fixed_exploit, label_fixed_reconna, label_fixed_shellode,
               label_fixed_ddos, label_fixed_analysis, label_fixed_backdoor, label_fixed_worms, label_fixed_generic]

test_fixed_data = np.r_[test_normal_fixed_data, test_fuzzer_fixed_data, test_exploit_fixed_data, test_reconna_fixed_data, test_shellcode_fixed_data,
                test_ddos_fixed_data, test_analysis_fixed_data, test_backdoor_fixed_data, test_worms_fixed_data, test_generic_fixed_data]

test_fixed_label = np.r_[test_label_fixed_normal, test_label_fixed_fuzzer, test_label_fixed_exploit, test_label_fixed_reconna, test_label_fixed_shellode,
    test_label_fixed_ddos, test_label_fixed_analysis, test_label_fixed_backdoor, test_label_fixed_worms, test_label_fixed_generic]

if train_fixed_label.shape[0] == train_fixed_data.shape[0]:
    if test_fixed_label.shape[0] == test_fixed_data.shape[0]:
        print('Outliers Remove Finished......')

print('正在保存数据....')
np.save('Isolation_train_data_8W.npy', train_fixed_data)
np.save('Isolation_train_label_8W.npy', train_fixed_label)
np.save('Isolation_test_data_17W0.1.npy', test_fixed_data)
np.save('Isolation_test_label_17W0.1.npy', test_fixed_label)

print('保存数据完毕...')
