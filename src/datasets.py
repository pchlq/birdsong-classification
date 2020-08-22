import numpy as np 
import pandas as pd
import random 
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transform import FrequencyMask, TimeMask
import config
from pathlib import Path


class BSImageData(Dataset):
    def __init__(self, data, val_fold=0, train=True, max_freqmask_width=15, max_timemask_width=15):
        self.data = data
        if train:
            files = self.data[self.data.fold != val_fold].reset_index(drop=True)
        else:
            files = self.data[self.data.fold == val_fold].reset_index(drop=True)
        
        self.items = files.im_path.values
        self.labels = files.ebird_code.values
        self.length = len(self.items)
        self.max_freqmask_width = max_freqmask_width
        self.max_timemask_width = max_timemask_width
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([FrequencyMask(self.max_freqmask_width)], p=0.4),
            transforms.RandomApply([TimeMask(self.max_timemask_width)], p=0.4),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
        ])
        
    def __getitem__(self, index):
        fname = self.items[index]
        label = self.labels[index]
        img = Image.open(fname)
        img = (np.array(img.convert('RGB')) / 255.).astype(np.float32)
        return (self.transforms(img), label)
            
    def __len__(self):
        return self.length



if __name__ == "__main__":

    df = pd.read_csv(config.TRAIN_DF)

    ebird_dct = {}
    for i, label in enumerate(df.ebird_code.unique()):
        ebird_dct[label] = i
    print(len(ebird_dct))
    # train_data = df.loc[:, ["im_path", "ebird_code", "fold"]]
    # train_data.ebird_code = train_data.ebird_code.map(ebird_dct)

    # im, lab = BSImageData(train_data)[0]
    # print(im)
    # print(lab)