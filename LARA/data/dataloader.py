from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


# 重构数据集
class dataset(Dataset):
    def __init__(self, train_csv, user_emb_matrix):
        self.train_csv = pd.read_csv(train_csv, header=None)
        self.user = self.train_csv.loc[:, 0]
        self.item = self.train_csv.loc[:, 1]
        self.attr = self.train_csv.loc[:, 2]
        self.user_emb_matrix = pd.read_csv(user_emb_matrix, header=None)
        self.user_emb_values = np.array(self.user_emb_matrix[:])

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        user_emb = self.user_emb_values[user]
        # user, item, attr, user_emb
        attr = self.attr[idx][1:-1].split()
        attr = torch.tensor(list([int(item) for item in attr]), dtype=torch.long)
        attr = np.array(attr)
        return user, item, attr, user_emb


def load_test_data():
    test_item = pd.read_csv('test/test_item.csv', header=None).loc[:]
    test_item = np.array(test_item)
    test_attribute = pd.read_csv('../../movie_all_code/ui_represent/test_attribute.csv', header=None).loc[:]
    test_attribute = np.array(test_attribute)
    return test_item, test_attribute