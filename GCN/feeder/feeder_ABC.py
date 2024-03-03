import numpy as np
import json
import random
import os
from torch.utils.data import Dataset

class Feeder(Dataset):
  def __init__(self, root_path):
    self.data = None
    self.labels = None
    self.root_path = root_path
    self.load_data()
    self.bone = [(0, 1), (0, 2), (2, 4), (1, 3), (4, 6), (3, 5), (6, 8), (5, 7), (8, 10), (7, 9), (6, 12), (5, 11), (6, 5), (12, 11)]
    self.time_steps = 60

  def load_data(self):
    self.data = []
    self.labels = []
    for file_name in os.listdir(self.root_path):
      data_path = os.path.join(self.root_path, file_name)
      with open(data_path, 'r') as f:
        json_file = json.load(f)
      skeletons = json_file['skeletons']
      label = json_file['label']
      value = np.array(skeletons)
      self.labels.append(label)
      self.data.append(value)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    label = self.labels[index]
    value = self.data[index]

    random.random()
    center = value[0, :]
    value = value - center

    value = np.reshape(value, (-1, 2))
    value = (value - np.min(value, axis=0)) / (np.max(value, axis=0) - np.min(value, axis=0))

    value = value * 2 - 1
    value = np.reshape(value, (-1, 13, 2))

    data = np.zeros((self.time_steps, 13, 2))

    length = value.shape[0]

    random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
    random_idx.sort()
    data[:, :, :] = value[random_idx, :, :]
    data[:, :, :] = value[random_idx, :, :]

    data_bone = np.zeros_like(data)
    for bone_idx in range(len(self.bone)):
        data_bone[:, self.bone[bone_idx][0], :] = data[:, self.bone[bone_idx][0], :] - data[:, self.bone[bone_idx][1], :]
    data = data_bone

    data = np.transpose(data, (2, 0, 1))
    C,T,V = data.shape
    data = np.reshape(data,(C,T,V,1))

    return data, label, index

