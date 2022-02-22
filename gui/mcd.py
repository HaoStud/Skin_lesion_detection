import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn.functional as F

from collections import Counter

from model.models import ImgClassificationModel
from model.monte_carlo import predict as md_pred

# params
checkpoint  = 'output/checkpoints/resnet-drop_epoch=13_val_acc=0.80.ckpt'
device      = torch.device('cuda:0')
classes     = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
num_classes = len(classes)

# model
model = ImgClassificationModel.load_from_checkpoint(checkpoint)
model = model.to(device)

# predict
def predict(X, T=100):
    X = X.to(device)
    preds = np.zeros((T, len(classes)))
    for t in range(T):
        pred = md_pred(model, X, dropout=True)
        pred = F.softmax(pred, dim=1)
        preds[t, :] = pred.cpu().detach().numpy()

    return preds

# plotting
def plot_preds(y_preds):
    df  = pd.DataFrame(y_preds, columns=classes)

    fig = plt.figure()
    fig.patch.set_alpha(0.)

    plt.title('Monte Carlo Prediction')
    plt.xlabel('Cancer Type')
    plt.ylabel('Probabilty')
    sns.boxplot(data=df, showfliers=False)

    return fig


def plot_preds_max(y_preds):
    pred_counter = Counter(np.argmax(y_preds, axis=1))
    pred_array   = [pred_counter[i]/100 for i in range(num_classes)]
    pred_array   = np.array(pred_array)
    df = pd.DataFrame(pred_array, index=classes).T

    fig = plt.figure()
    fig.patch.set_alpha(0.)

    plt.title('Monte Carlo Prediction')
    plt.xlabel('Cancer Type')
    plt.ylabel('Probabilty')
    sns.barplot(data=df)

    return fig
