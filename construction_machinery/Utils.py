import os
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.metrics import f1_score

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CustomDataset(Dataset):
    def __init__(self, kind='train', sampling='no'):
        super(CustomDataset, self).__init__()
        data_root_path = '/data/jyhwang/construction_machinery/Data/sampling'

        if kind == 'train':
            data_root_path = os.path.join(data_root_path, 'Train', sampling)
        else:
            data_root_path = os.path.join(data_root_path, 'Validation')

        self.data_x = np.load(os.path.join(data_root_path, 'x.npy'))
        self.data_y = np.load(os.path.join(data_root_path, 'y.npy'))
        self.s_data_x = np.load(os.path.join(data_root_path, 's_x.npy'))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        teacher_X = torch.Tensor(self.data_x[index])
        student_X = torch.Tensor(self.s_data_x[index])
        y = self.data_y[index]

        return teacher_X, student_X, y

class CRD_Contrast_Dataset(Dataset):
    def __init__(self, sampling='no'):
        super(CRD_Contrast_Dataset, self).__init__()
        data_root_path = '/data/jyhwang/construction_machinery/Data/sampling/Train/'
        data_root_path = os.path.join(data_root_path, sampling)

        self.k = 10

        self.data_x = np.load(os.path.join(data_root_path, 'x.npy'))
        self.data_y = np.load(os.path.join(data_root_path, 'y.npy'))
        self.s_data_x = np.load(os.path.join(data_root_path, 's_x.npy'))

        num_samples = len(self.data_x)
        num_classes = 2

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[self.data_y[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        teacher_X = torch.Tensor(self.data_x[index])
        student_X = torch.Tensor(self.s_data_x[index])
        y = self.data_y[index]

        pos_idx = index
        replace = True if self.k > len(self.cls_negative[y]) else False
        neg_idx = np.random.choice(self.cls_negative[y], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        return teacher_X, student_X, y, index, sample_idx

class Hyperparameters:
    def __init__(self):
        self.str_hyper_parameter_list = []
        self.hyperparameter_list = []

        # Hyperparameters
        for lr in [1e-4, 1e-3]:
            for min_lr in [1e-8, 1e-7]:
                for reg in [1e-3, 0]:
                    for dimension in [[256, 1024, 1024, 256], [256, 1024, 256], [128, 512, 512, 128], [128, 512, 128]]:
                        self.hyperparameter_list.append({'lr': lr, 'min_lr': min_lr, 'reg': reg, 'dimension': dimension})
                        self.str_hyper_parameter_list.append(str(lr) + '_' + str(min_lr) + '_' + str(reg) + '_' + str(dimension))

    # Save Result
    def save_val_test(self, result, file_name, hyperparameter_list, hyperparameter):
        save_root_path = os.path.join('./Result/Performance')
        createFolder(save_root_path)

        # Save Path
        save_path = os.path.join(os.path.join(save_root_path), file_name + '.csv')

        col = ['T-F1', 'S-F1', 'Final']
        index = hyperparameter_list

        if not os.path.isfile(save_path):
            result_df = pd.DataFrame(columns=col, index=index)
            result_df.to_csv(save_path)

        result_df = pd.read_csv(save_path, index_col=0)
        for c in col:
            result_df.loc[hyperparameter, c] = result[c]
        result_df.to_csv(save_path)