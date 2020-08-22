from pathlib import Path
import pandas as pd
import torch


bs = 32
num_epochs = 5
lr = 0.0001
mix_precision = True
opt_level = 'O1'
device = 'cuda'

ALL_FOLDS = Path("../spect_images")
TRAIN_DF = ALL_FOLDS / "train_all_new.csv"
# print( len(list(ALL_FOLDS.glob("fold*/*/*.png"))) )

# df = pd.read_csv(TRAIN_DF)
