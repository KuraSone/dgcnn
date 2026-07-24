"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: dataset.py
@Time: 2018/10/13 6:21 PM
"""

import glob
import os
from typing import Iterator

import h5py
import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset


def download() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], data_dir))
        os.system('rm %s' % zipfile)


def load_data(partition: str) -> tuple[ndarray, ndarray]:
    download()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    data_list = []
    label_list = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        raw_data = f['data'][:].astype('float32')
        raw_label = f['label'][:].astype('int64')
        f.close()
        data_list.append(raw_data)
        label_list.append(raw_label)
    all_data = np.concatenate(data_list, axis=0)
    all_label = np.concatenate(label_list, axis=0)
    return all_data, all_label


def translate_point_cloud(point_cloud: ndarray) -> ndarray:
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    return np.asarray(np.add(np.multiply(point_cloud, xyz1), xyz2), dtype=np.float32)


def jitter_point_cloud(point_cloud: ndarray, sigma: float = 0.01, clip: float = 0.02) -> ndarray:
    num_points, num_channels = point_cloud.shape
    point_cloud += np.clip(sigma * np.random.randn(num_points, num_channels), -1 * clip, clip)
    return point_cloud


class ModelNet40(Dataset[tuple[ndarray, ndarray]]):
    def __init__(self, num_points: int, partition: str = 'train') -> None:
        self.data: ndarray
        self.label: ndarray
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item: int) -> tuple[ndarray, ndarray]:
        point_cloud = self.data[item][:self.num_points]
        label_ = self.label[item]
        if self.partition == 'train':
            point_cloud = translate_point_cloud(point_cloud)
            np.random.shuffle(point_cloud)
        return point_cloud, label_

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __iter__(self) -> Iterator[tuple[ndarray, ndarray]]:
        for i in range(len(self)):
            yield self[i]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
