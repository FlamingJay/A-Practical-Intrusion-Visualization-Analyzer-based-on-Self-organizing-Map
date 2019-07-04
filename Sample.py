'本程序将对数据进行上下采样'
print(__doc__)

import numpy as np
import pandas as pd

train_data = np.load('Isolation_train_data_8W.npy')
train_label = np.load('Isolation_train_label_8W.npy')

normal = train_data[np.where(train_label == 0)[0], :]
df_normal_data = pd.DataFrame(normal)
normal = np.array(df_normal_data.sample(frac=0.2, replace=False))
label_normal = np.repeat(0, len(normal))

Fuzzer = train_data[np.where(train_label == 1)[0], :]
label_fuzzer = np.repeat(1, len(Fuzzer))

Exploit = train_data[np.where(train_label == 2)[0], :]
label_exploit = np.repeat(2, len(Exploit))

Recona = train_data[np.where(train_label == 3)[0], :]
label_reconna = np.repeat(3, len(Recona))

Shellcode = train_data[np.where(train_label == 4)[0], :]
label_shellcode = np.repeat(4, len(Shellcode))

Dos = train_data[np.where(train_label == 5)[0], :]
label_ddos = np.repeat(5, len(Dos))

Analysis = train_data[np.where(train_label == 6)[0], :]
label_analysis = np.repeat(6, len(Analysis))

Backdoor = train_data[np.where(train_label == 7)[0], :]
label_backdoor = np.repeat(7, len(Backdoor))

Worrms = train_data[np.where(train_label == 8)[0], :]
label_worms = np.repeat(8, len(Worrms))

Generic = train_data[np.where(train_label == 9)[0], :]
df_generic_data = pd.DataFrame(Generic)
Generic = np.array(df_generic_data.sample(frac=0.1, replace=False))
label_generic = np.repeat(9, len(Generic))

train_fixed_data = np.r_[normal, Fuzzer, Exploit, Recona, Shellcode, Dos, Analysis, Backdoor, Worrms, Generic]

train_fixed_label = np.r_[label_normal, label_fuzzer, label_exploit, label_reconna, label_shellcode,
               label_ddos, label_analysis, label_backdoor, label_worms, label_generic]

np.save('sample2_train_data.npy', train_fixed_data)
np.save('sample2_train_label.npy', train_fixed_label)
print(1)