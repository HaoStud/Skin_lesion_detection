import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob

import torch
import torch.nn.functional as F

from model.models import ImgClassificationModel

# models to load
model_names    = ['resnet-drop', 'resnet', 'alexnet', 'vgg', 'densenet', ] # 'squeezenet', ]
checkpoint_dir = 'output/checkpoints'
checkpoints    = {m: glob(f'{checkpoint_dir}/{m}_*')[0] for m in model_names}

# load models
device = torch.device('cuda:0')
models = {m: ImgClassificationModel.load_from_checkpoint(checkpoints[m]).to(device)
          for m in model_names}

# params
classes    = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
n_classes  = len(classes)
n_models   = len(models)


# predict
def predict(X):
    X = X.to(device)
    preds = {}
    for name, model in models.items():
        with torch.no_grad():
            model.eval()
            pred = model(X)
        pred = F.softmax(pred, dim=1)
        preds[name] = pred.cpu().detach().numpy()[0]

    # mean
    pred_mean     = np.array(list(preds.values())).mean(axis=0)
    preds['mean'] = pred_mean

    return preds

def ens_predict(X):
    X = X.to(device)
    preds = {}
    for name, model in models.items():
        with torch.no_grad():
            model.eval()
            pred = model(X)
        pred = F.softmax(pred, dim=1)
        preds[name] = pred.cpu().detach().numpy()[0]

    # mean
    pred_mean     = np.array(list(preds.values())).mean(axis=0)
    preds['mean'] = pred_mean

    return np.argmax(preds['mean'])


# plotting
def plot_preds(y_preds):
    df  = pd.DataFrame(y_preds, index=classes)

    df_1 = df.loc[:, df.columns != "mean"]
    df_2 = df.loc[:, df.columns == "mean"]

    fig = sns.relplot(data=df_1).fig
    sns.barplot(data=df_2.T, alpha=0.2, color='blue')
    plt.title('Ensemble Prediction')
    plt.xlabel('Cancer Type')
    plt.ylabel('Probabilty')
    fig.patch.set_alpha(0.)

    return fig
