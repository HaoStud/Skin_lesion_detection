#!/usr/bin/python3

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from layer.relu_dropout import ReLUDropout


def add_dropout_layers(model, base_rate=0.1):
    global drop_idx
    drop_idx = 0
    add_dropout_after_relu(model, base_rate)


def add_dropout_after_relu(model, base_rate):
    """
    Adds dropout after ReLu, 
    Dropout rate: p = baserate + 0.01*drop_idx
    The drop_idx increments by 1 after each added dropout layer
    """

    global drop_idx
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            add_dropout_after_relu(module, base_rate)
        if isinstance(module, nn.ReLU):
            # new = nn.Sequential(module, nn.Dropout2d(p=base_rate + 0.01 * drop_idx, inplace=True))  # does not work, cant backprop two inplace ops
            new = ReLUDropout(inplace=True, p=base_rate)  # + 0.01 * drop_idx)
            setattr(model, name, new)
            drop_idx += 1


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if "Dropout" in m.__class__.__name__:
            m.train()


def predict(model, X, dropout=True):
    with torch.no_grad():
        model.eval()
        if dropout:
            enable_dropout(model)
        return model(X)


def predict_mcd_proba(model, X, num_samples):
    preds = None
    for _ in range(num_samples):
        pred = predict(model, X, dropout=True)
        pred = F.softmax(pred, dim=1)

        if preds == None: preds = pred
        else:             preds += pred
    return preds / num_samples


def predict_mcd_class(model, X, num_samples):
    proba_preds = predict_mcd_proba(model, X, num_samples)
    proba_preds = proba_preds.cpu().detach().numpy()
    return np.argmax(proba_preds, axis=1)
