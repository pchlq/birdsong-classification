from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from PIL import Image
from typing import Union
from handlers import AverageMeter
import config
import torch
from torch import Tensor
import apex as amp


def to_numpy(tensor: Union[Tensor, Image.Image, np.array]) -> np.ndarray:
    
    if type(tensor) == np.array or type(tensor) == np.ndarray:
        return np.array(tensor)
    elif type(tensor) == Image.Image:
        return np.array(tensor)
    elif type(tensor) == Tensor:
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError(msg)


def train_fn(data_loader, model, criterion, optimizer, epoch, config, device=config.device):

    # y_true_train, y_pred_train = [], []

    model.train()
    loss_handler = AverageMeter()
    score_handler = AverageMeter()
    # f1_metric = F1Score("micro")

    pbar = tqdm(total=len(data_loader) * config.bs)
    pbar.set_description(
        " Epoch {}, lr: {:.2e}".format(epoch + 1, get_learning_rate(optimizer))
    )

    for i, (inputs, target) in enumerate(data_loader):

        optimizer.zero_grad()
        inputs = inputs.to(device)
        target = target.to(device)
        out = model(inputs)

        loss = criterion(out, target)

        if config.mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        optimizer.step()

        if config.step_scheduler:
            scheduler.step()

        loss_handler.update(loss.item())                         
        score_handler.update( f1_score(to_numpy(target), to_numpy(torch.argmax(out, dim=1)), average='micro') )

        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_lr = get_learning_rate(optimizer)
        batch_size = len(inputs)
        pbar.update(batch_size)
        pbar.set_postfix(loss=f"{loss_handler.avg:.5f}")

        # f1_new = f1_metric( torch.argmax(out, dim=1), target )
        # new_score.append(f1_new.item())

    pbar.close()
    
    return loss_handler, score_handler.avg


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def eval_fn(data_loader, model, criterion, device=config.device):
    model.eval()
    loss_handler = AverageMeter()
    score_handler = AverageMeter()
    # num_correct = 0
    # num_examples = 0
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for inputs, target in tk0:

            inputs = inputs.to(device)
            target = target.to(device)
            out = model(inputs)
            loss = criterion(out, target)

            loss_handler.update(loss.item())
            score_handler.update( f1_score(to_numpy(target), to_numpy(torch.argmax(out, dim=1)), average='micro') )
            # accuracy_handler.update( to_numpy(torch.eq(torch.argmax(out, dim=1), target)) )
            # correct = torch.eq(torch.argmax(torch.softmax(out, dim=1), dim=1), target).view(-1)
            # num_correct += torch.sum(correct).item()
            # num_examples += correct.shape[0]

            tk0.set_postfix(loss=f"{loss_handler.avg:.5f}")

    return loss_handler, score_handler.avg #num_correct / num_examples
