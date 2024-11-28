import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

TRAIN_PATH = './data/train_data/'
VALID_PATH = './data/valid_data/'
TEST_PATH = './data/test.csv'


class AdversarialDataset(Dataset):
    def __init__(self, mode='train'):
        '''
        mode: 'train' ,'valid', 'test'
        train_type: 'smote', 'adasyn', 'smoteenn'
        '''
        self.mode = mode
        # self.train_type = train_type

        if self.mode == 'train':
            self.df = pd.read_csv(TRAIN_PATH + 'smote.csv')
        elif self.mode == 'valid':
            self.df = pd.read_csv(VALID_PATH + 'smote.csv')
        elif self.mode == 'test':
            self.df = pd.read_csv(TEST_PATH)

        df_features = self.df.drop(['Diagnosis', 'Gender', '진단시점나이'], axis=1)
        df_labels = self.df['Diagnosis']

        self.feature_names = self.get_feature_names(df_features)
        self.features = df_features.values.astype(np.float32)

        self.labels = df_labels.values
        self.labels_for_ARN = np.where(self.labels == 1, 1, 0)
        self.labels_for_CMV = np.where(self.labels == 2, 1, 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label_for_adv = self.labels[idx]
        label_for_ARN = self.labels_for_ARN[idx]
        label_for_CMV = self.labels_for_CMV[idx]
        return feature, (label_for_ARN, label_for_CMV, label_for_adv)

    def get_feature_names(self, df):
        original_feature_names = list(df.columns)

        # Remove '[...]'
        feature_names = list(
            map(lambda x: x.split('[')[0], original_feature_names))

        # Change 'WBC COUNT' to 'WBC(#)'
        feature_names = list(
            map(lambda x: x.replace(' COUNT', '(#)'), feature_names))
        return feature_names
