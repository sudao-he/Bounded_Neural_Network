import numpy as np
import os
import time
from collections import Counter


class FBGDataGenerator():
    def __init__(self, data_path, skiprows=105, channels_num=range(2+16, 2+16+6), downsampling=1, test_time_interval=[],
                 time_keep=True, sampling_rate=5000, sample_len=1, seq_len=1, step_len=1):
        super(FBGDataGenerator, self).__init__()
        T1 = time.time()
        files_paths = FBGDataGenerator.find_file_name(data_path)
        self.downsampling = downsampling
        self.time_interval = test_time_interval
        self.sampling_rate = sampling_rate
        self.sample_len = sample_len
        self.seq_len = seq_len
        self.step_len = step_len
        self.time_keep = time_keep
        # i = 0
        self.data = np.empty((0, 6))
        self.time_stamp_datetime = np.array([], dtype='datetime64[us]')
        for i in range(0, len(files_paths)):
            files_path = files_paths[i]
            print('loading from txt:', files_path)
            self.data_tem = np.loadtxt(files_path, skiprows=skiprows, usecols=channels_num)
            self.data = np.vstack((self.data, self.data_tem))
            self.time_stamp = np.loadtxt(files_path, dtype='str', skiprows=skiprows, usecols=[0, 1])
            self.time_stamp = self.datetime_convert(self.time_stamp)
            self.time_stamp_datetime = np.append(self.time_stamp_datetime, self.time_stamp)
                # print(i)
            print('Processing: {}/{}'.format(i+1, len(files_paths)))
        T2 = time.time()
        print('processing time:%s' % (T2-T1))
        print('Data shape:', self.data.shape)
        print('time shape:', self.time_stamp_datetime.shape)

        # Reshape data
        self.label = np.empty((0))
        self.sensor_number = np.empty((0))
        self.time = np.array([], dtype='datetime64[s]')
        self.reshaped_data = np.empty((0, int(sample_len * sampling_rate / downsampling)))
        self.reshaped_data_seq, self.label, self.sensor_number, self.time = self.dateset_generator()

        print('Samples:', self.reshaped_data_seq.shape)
        print('Labels:', self.label.shape)
        print('Sensor_num:', self.sensor_number.shape)
        print('Time:', self.time.shape)
        print('Sensor Num distribution:', Counter(self.sensor_number))
        print('Class distribution:', Counter(self.label))
        # Save data

        data_path = '../data_utils/data-ds_{:d}-sample_len_{:d}.npy'.format(self.downsampling, self.sample_len)
        time_path = '../data_utils/time-ds_{:d}-sample_len_{:d}.csv'.format(self.downsampling, self.sample_len)
        label_path = '../data_utils/label-ds_{:d}-sample_len_{:d}.csv'.format(self.downsampling, self.sample_len)
        type_path = '../data_utils/type-ds_{:d}-sample_len_{:d}.csv'.format(self.downsampling, self.sample_len)

        np.save(data_path, self.reshaped_data_seq)
        np.savetxt(time_path, self.time, fmt='%s', delimiter=',')
        np.savetxt(label_path, self.label, delimiter=',')
        np.savetxt(type_path, self.sensor_number, delimiter=',')


    def dateset_generator(self):
        # segmentation & downsampling
        time_stamp_datetime = np.array(self.time_stamp_datetime, dtype='datetime64[s]')
        time_interval = np.array(self.time_interval, dtype='datetime64[s]')
        time_interval_list = []
        for i in range(len(self.time_interval)):
            time_interval_row = np.where(time_stamp_datetime == time_interval[i])[0]
            if time_interval_row.size != 0:
                time_interval_list.append(time_interval_row[0])
                print('Find data at row {}'.format(time_interval_row[0]))
                print('Data time: {}'.format(np.timedelta64(time_interval[i] - self.time_stamp_datetime[0], 's')))
            else:
                print('Data missing at time {}'.format(np.datetime_as_string(time_interval[i], unit='s')))
        masks = [not self.time_keep] * self.data.shape[0]
        for i in range(0, len(time_interval_list), 2):
            masks[time_interval_list[i]: time_interval_list[i + 1]] = [self.time_keep] * (
                        time_interval_list[i + 1] - time_interval_list[i])
        data = self.data[0:len(masks):self.downsampling, :]
        masks = masks[0:len(masks):self.downsampling]
        masked_data = data[masks, :]
        masked_time_stamp = self.time_stamp_datetime[masks]
        begin_time = np.array(masked_time_stamp.min(), dtype='datetime64[s]') + np.timedelta64(1, 's')
        end_time = np.array(masked_time_stamp.max(), dtype='datetime64[s]')
        second_count = np.timedelta64(end_time - begin_time, 's') + 1
        step_len = np.timedelta64(self.step_len, 's')
        sample_len_datetime = np.timedelta64(self.sample_len, 's')
        steps = int((second_count - sample_len_datetime) / step_len + 1)
        seq_len = int(self.seq_len * self.sampling_rate / self.downsampling)
        seq_num = int(self.sample_len / self.seq_len)
        sensor_num = np.array([0, 1, 2, 3, 4, 5])
        for i in range(steps):
            sample = masked_data[np.where((masked_time_stamp >= begin_time) & (masked_time_stamp < begin_time + sample_len_datetime))]
            if sample.shape[0] >= self.sample_len * self.sampling_rate / self.downsampling:
                sample_reshape = sample[:int(self.sample_len * self.sampling_rate / self.downsampling)].T
                if begin_time >= time_interval[0] and begin_time <= time_interval[1]:
                    label = np.array([1] * 6)
                elif begin_time >= time_interval[6] and begin_time <= time_interval[7]:
                    label = np.array([2] * 6)
                else:
                    label = np.array([0] * 6)
                self.reshaped_data = np.vstack((self.reshaped_data, sample_reshape))
                self.sensor_number = np.append(self.sensor_number, sensor_num)
                self.time = np.append(self.time, begin_time)
                self.label = np.append(self.label, label)
            begin_time += step_len
        self.reshaped_data_seq = np.reshape(self.reshaped_data, (self.reshaped_data.shape[0], seq_num, seq_len))
        return self.reshaped_data_seq, self.label, self.sensor_number, self.time


    def datetime_convert(self, time_stamp):
        list1 = time_stamp[:, 0].tolist()
        list2 = time_stamp[:, 1].tolist()
        list1 = list(map(format_time, list1))
        list_time = [x1 + 'T' + x2 for x1, x2 in zip(list1, list2)]
        time_stamp_datetime = np.array(list_time, dtype='datetime64[us]')
        return time_stamp_datetime

    @staticmethod
    def find_file_name(rootdir, file_type='.txt'):
        L = []
        for root, dirs, files in os.walk(rootdir):
            # print('root=', root, ',dirs=', dirs, ',files=', files,)
            for file in files:
                filename = os.path.splitext(file)
                if filename[1] == file_type:
                    #                print('file=', filename[0])
                    dir = os.path.split(file)
                    root_dir = os.path.join(root, dir[1])
                    L.append(root_dir)
        #    print('L=',L)
        return L


def format_time(x):
    year = x[0:4]
    fisrt_hyphen = x.find('/')
    second_hyphen = x.find('/', fisrt_hyphen + 1)
    month = x[fisrt_hyphen + 1:second_hyphen]
    day = x[second_hyphen + 1:]
    time_str = f'{int(year):d}-{int(month):02d}-{int(day):02d}'
    return time_str


if __name__ == '__main__':
    break_time_list = ['2022-12-05T17:55:00', '2022-12-05T18:01:00', '2022-12-05T18:28:00',
                       '2022-12-05T18:49:00', '2022-12-05T19:00:00', '2022-12-05T20:00:00',
                       '2022-12-05T20:08:00', '2022-12-05T20:43:00']
    a = FBGDataGenerator(data_path='YourPath',
                   test_time_interval=break_time_list)