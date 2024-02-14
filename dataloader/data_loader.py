import torch.utils.data as data
import numpy as np
import os
import time
from collections import Counter
from sklearn import metrics


class FBGDataset(data.Dataset):
    """
    data_path:文件路径
    label_path:标签路径
    sensor_num_path:传感器标号路径
    time_path:样本记录时间路径
    """

    def __init__(self, data_path='../data_utils/data-ds_1-sample_len_6.npy',
                 label_path='../data_utils/label-ds_1-sample_len_6.csv',
                 time_path='../data_utils/time-ds_1-sample_len_6.csv',
                 sensor_num_path='../data_utils/type-ds_1-sample_len_6.csv', feature_length=5000, reshape=True):
        super(FBGDataset, self).__init__()
        self.raw_data = np.load(data_path)
        self.data_reshape = self.raw_data.reshape((-1, self.raw_data.shape[1] * self.raw_data.shape[2]))
        self.labels = np.loadtxt(label_path)
        self.time = np.loadtxt(time_path, dtype='datetime64[us]')
        self.sensor_num = np.loadtxt(sensor_num_path)
        self.time = np.repeat(self.time, self.sensor_num.max() + 1)
        self.norm_data = np.zeros((self.data_reshape.shape))
        self.reshape = reshape
        for i in range(int(self.sensor_num.max() + 1)):
            self.norm_data[self.sensor_num == i, :] = FBGDataset.z_score(self.data_reshape[self.sensor_num == i, :])
        if reshape:
            self.data = self.norm_data.reshape((-1, int(self.data_reshape.shape[1] / feature_length),
                                                feature_length))
        else:
            self.data = self.norm_data
        print('Samples:', self.data.shape)
        print('Labels:', self.labels.shape)
        print('Sensor_num:', self.sensor_num.shape)
        print('Time:', self.time.shape)
        print('Sensor Num distribution:', Counter(self.sensor_num))
        print('Class distribution:', Counter(self.labels))

    def __getitem__(self, index):
        labels = self.labels[index]
        if self.reshape:
            data = self.data[index, :, :]
        else:
            data = self.data[index, :]
#        sensor_num = self.sensor_num[index]
#        time = self.time[index]
#        sample = {'data': data, 'label': labels, 'time': time, 'sensor_num': sensor_num}
        sample = {'data': data, 'label': labels}
        return sample

    def z_score(x):
        mean_x = x.mean()
        std_x = x.std()
        # print(mean_x)
        # print(std_x)
        x = (x - mean_x) / std_x
        return x
        # 1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        # 2。预处理数据（例如torchvision.Transform）。
        # 3。返回数据对（例如图像和标签）。
        # 这里需要注意的是，第一步：read one data，是一个data

    def __len__(self):
        return self.labels.shape[0]


class FBGNoisyDataset(data.Dataset):
    """
    data_path:文件路径
    label_path:标签路径
    sensor_num_path:传感器标号路径
    time_path:样本记录时间路径
    """

    def __init__(self, data_path='../data_utils/data-ds_1-sample_len_6.npy',
                 label_path='../data_utils/label-ds_1-sample_len_6.csv',
                 time_path='../data_utils/time-ds_1-sample_len_6.csv',
                 sensor_num_path='../data_utils/type-ds_1-sample_len_6.csv', feature_length=5000, reshape=True,
                 noise_ratio=0.4):
        super(FBGNoisyDataset, self).__init__()
        self.raw_data = np.load(data_path)
        self.noise_ratio = noise_ratio
        self.data_reshape = self.raw_data.reshape((-1, self.raw_data.shape[1] * self.raw_data.shape[2]))
        self.labels = np.loadtxt(label_path)
        self.time = np.loadtxt(time_path, dtype='datetime64[us]')
        self.sensor_num = np.loadtxt(sensor_num_path)
        self.time = np.repeat(self.time, self.sensor_num.max() + 1)
        self.norm_data = np.zeros((self.data_reshape.shape))
        self.reshape = reshape
        for i in range(int(self.sensor_num.max() + 1)):
            self.norm_data[self.sensor_num == i, :] = FBGDataset.z_score(self.data_reshape[self.sensor_num == i, :])
        if reshape:
            self.data = self.norm_data.reshape((-1, int(self.data_reshape.shape[1] / feature_length),
                                                feature_length))
        else:
            self.data = self.norm_data
        self.noisy_labels = self.labels.copy()
        self.count = range(self.labels.shape[0])

        """
        Randomly change the labels
        """

        self.uniform_noise_injection()

        """
        Get relative time 
        """
        # self.time_min = self.time.min()
        # self.time_shift = self.time - self.time_min
        # self.time_shift_second = self.time_shift.astype('timedelta64[s]')
        # self.time_shift_second_int = self.time_shift_second.astype(int)
        self.relative_time = np.arange(int(self.sensor_num.shape[0] / (self.sensor_num.max() + 1)))
        self.relative_time = np.repeat(self.relative_time, self.sensor_num.max() + 1)


        print('Samples:', self.data.shape)
        print('Labels:', self.labels.shape)
        print('Sensor_num:', self.sensor_num.shape)
        print('Time:', self.time.shape)
        print('Sensor Num distribution:', Counter(self.sensor_num))
        print('Class distribution:', Counter(self.labels))

    def uniform_noise_injection(self):
        num = int(self.noise_ratio * self.labels.shape[0])
        noisy_index = np.random.choice(self.labels.shape[0], size=num, replace=False)
        noise = np.random.randint(low=1, high=self.labels.max() + 1, size=num)
        self.noisy_labels[noisy_index] = self.labels[noisy_index] + noise
        self.noisy_labels[self.noisy_labels > self.labels.max()] = self.noisy_labels[self.noisy_labels > self.labels.max()] - self.labels.max() - 1
        ratio = metrics.accuracy_score(self.labels, self.noisy_labels)
        print('Noise ratio after injection: {:2f}'.format(1-ratio))

    def __getitem__(self, index):
        labels = self.labels[index]
        noisy_labels = self.noisy_labels[index]
        if self.reshape:
            data = self.data[index, :, :]
        else:
            data = self.data[index, :]
        sensor_num = self.sensor_num[index]
        time = self.relative_time[index]
#        sample = {'data': data, 'label': labels, 'time': time, 'sensor_num': sensor_num}
        count_index = self.count[index]
        sample = {'data': data, 'label': labels, 'noisy_labels': noisy_labels, 'index': count_index,
                  'time': time, 'sensor_num': sensor_num}
        return sample

    def z_score(x):
        mean_x = x.mean()
        std_x = x.std()
        # print(mean_x)
        # print(std_x)
        x = (x - mean_x) / std_x
        return x
        # 1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        # 2。预处理数据（例如torchvision.Transform）。
        # 3。返回数据对（例如图像和标签）。
        # 这里需要注意的是，第一步：read one data，是一个data

    def __len__(self):
        return self.labels.shape[0]

if __name__ == '__main__':
    ds = 1
    sample_len = 1
    data_path = '../data_utils/data-ds_{:d}-sample_len_{:d}.npy'.format(ds, sample_len)
    time_path = '../data_utils/time-ds_{:d}-sample_len_{:d}.csv'.format(ds, sample_len)
    label_path = '../data_utils/label-ds_{:d}-sample_len_{:d}.csv'.format(ds, sample_len)
    type_path = '../data_utils/type-ds_{:d}-sample_len_{:d}.csv'.format(ds, sample_len)
    MyFBGDataset1 = FBGNoisyDataset(data_path=data_path, time_path=time_path, label_path=label_path, sensor_num_path=type_path,
                                    feature_length=1000, noise_ratio=0)
    # norm_1 = np.zeros((MyFBGDataset1.raw_data.shape))
    # for i in range(int(MyFBGDataset1.sensor_num.max())):
    #     norm_1[MyFBGDataset1.sensor_num == i, :, :] = FBGDataset.z_score(MyFBGDataset1.raw_data[MyFBGDataset1.sensor_num == i, :, :])
    # norm_2 = MyFBGDataset1.data
    # error = norm_1 - norm_2
    print(MyFBGDataset1.__len__())
    print(MyFBGDataset1.__getitem__(11))