import os
import numpy as np 
import pandas as pd
import random 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import apex as amp

from datasets import BSImageData
from models import build_model
from engine import train_fn, eval_fn
import config


def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main(config):

    seed_everything(100)

    df = pd.read_csv(config.TRAIN_DF)
    ebird_dct = {}
    for i, label in enumerate(df.ebird_code.unique()):
        ebird_dct[label] = i

    train_data = df.loc[:, ["im_path", "ebird_code", "fold"]]
    train_data.ebird_code = train_data.ebird_code.map(ebird_dct)

    train_ds = BSImageData(train_data)
    val_ds = BSImageData(train_data, train=False)

    train_dl = DataLoader(train_ds, batch_size=config.bs, num_workers=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.bs, num_workers=4, shuffle=True)


    device = config.device
    model = build_model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.mix_precision:
        model, optimizer = amp.initialize(
                model, optimizer, opt_level=config.opt_level
        )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        train_loss, train_score = train_fn(
            train_dl, model, criterion, optimizer, config=config, epoch=epoch
        )

        valid_loss, valid_score, valid_acc = eval_fn(val_dl, model, criterion)

        print(
            f"|EPOCH {epoch+1}| F1_train {train_score:.5f}| F1_valid {valid_score:.5f}| Accuracy_valid {valid_acc:.3f}"
        )

        # logging.info(
        #     f"|EPOCH {epoch+1}| TRAIN_LOSS {train_loss.avg}| VALID_LOSS {valid_loss.avg}|"
        # )

        # if valid_loss.avg < best_loss:
        #     best_loss = valid_loss.avg
        #     print(f"New best model in epoch {epoch+1}")
        #     torch.save(model.state_dict(), project.checkpoint_dir / f"detr_best_{fold}.pth")

if __name__ == "__main__":
    main(config)