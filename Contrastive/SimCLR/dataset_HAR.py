from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import argparse
from utils import yaml_config_hook

parser = argparse.ArgumentParser(description="SimCLR")
config = yaml_config_hook("config/HAR_config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()

print("Dataset:", args.dataset)
# load data
args.dataset_train = torch.load('../../Datasets/HAR/train.pt')
print("Train Sample: {}; Label: {}.".format(args.dataset_train['samples'].shape, args.dataset_train['labels'].shape))

args.dataset_val = torch.load('../../Datasets/HAR/val.pt')
print("Val Sample: {}; Label: {}.".format(args.dataset_val['samples'].shape, args.dataset_val['labels'].shape))

args.dataset_test = torch.load('../../Datasets/HAR/test.pt')
print("Test Sample: {}; Label: {}.".format(args.dataset_test['samples'].shape, args.dataset_test['labels'].shape))

train_x = args.dataset_train['samples']
train_y = args.dataset_train['labels']

val_x = args.dataset_val['samples']
val_y = args.dataset_val['labels']

test_x = args.dataset_test['samples']
test_y = args.dataset_test['labels']

"""HAR train:
   There are 979 samples of Class 0
   There are 873 samples of Class 1
   There are 780 samples of Class 2
   There are 1024 samples of Class 3
   There are 1098 samples of Class 4
   There are 1127 samples of Class 5"""

'''balance train dataset'''
n_class = args.n_class
y = train_y.unsqueeze(1).unsqueeze(2)
extended_y = y.repeat(1, args.n_channel, 1)


traindataset = torch.cat((train_x, extended_y), 2)
traindataset = traindataset.numpy()

for i in range(n_class):
    print("There are {} samples of Class {}".format((train_y == i).sum(), i))
    id = train_y == i

    locals()['data'+str(i)] = traindataset[id]
    np.random.shuffle(locals()['data'+str(i)])
n_sample_perclass = 979  # 5880/6=980

downsampled_0 = data0[:n_sample_perclass]

data1 = np.repeat(data1, 2, axis=0)
downsampled_1 = data1[:n_sample_perclass]

data2 = np.repeat(data2, 2, axis=0)
downsampled_2 = data2[:n_sample_perclass]

downsampled_3 = data3[:n_sample_perclass]

downsampled_4 = data4[:n_sample_perclass]
downsampled_5 = data5[:n_sample_perclass]

traindataset = np.concatenate((downsampled_0, downsampled_1,
                               downsampled_2, downsampled_3,
                               downsampled_4, downsampled_5
                               ), axis=0)

n_seg = traindataset.shape[0]
print("After balancing, there are {} samples in total.".format(n_seg))
np.random.shuffle(traindataset)

n_length = args.n_length
n_feature = 206
train_x = traindataset[:, :, :n_length]

def zscore_norm(x):
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    train_x_norm = x.copy()
    for i in range(x.shape[1]):
        x_channel = x[:, i, :]
        scaler = StandardScaler()
        x_channel_norm = scaler.fit_transform(x_channel)
        train_x_norm[:, i, :] = x_channel_norm

    return train_x_norm

train_x = zscore_norm(train_x)
train_y = traindataset[:, :, n_feature:n_feature+1].mean(axis=1).squeeze(-1)

# z score val and test data
val_x = val_x[:, :, :n_length]
val_x = zscore_norm(val_x)

test_x = test_x[:, :, :n_length]
test_x = zscore_norm(test_x)
