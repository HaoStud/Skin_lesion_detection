#!/bin/env pipenv-shebang

import seaborn as sns
import numpy as np
import gradio as gr

from PIL import Image
from torchvision import transforms

import gui.mcd as mcd
import gui.ensemble as ens

def predictions(image):
    # transform input to tensor and unsqueeze (CNN expects 4th dim [batch])
    X = Image.fromarray(image.astype('uint8'), 'RGB')
    X = transforms.ToTensor()(X)
    X = X.unsqueeze(0)

    # monte carlo
    mcd_preds = mcd.predict(X)
    # mcd_plot  = mcd.plot_preds(mcd_preds)
    mcd_plot  = mcd.plot_preds_max(mcd_preds)

    # ensemble
    ens_pred = ens.predict(X)
    ens_plot = ens.plot_preds(ens_pred)

    # mean pred
    classes   = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    pred      = {c: float(ens_pred['mean'][i]) for (i,c) in enumerate(classes)}

    return pred, ens_plot, mcd_plot


if __name__ == '__main__':
    # set plotting theme
    sns.set_theme(); sns.set_style("whitegrid")

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    categories = [f"{lesion_type_dict[x]} ({x})" for x in sorted(lesion_type_dict.keys())]
    categories = ', '.join(categories)

    # gradio gui
    label = gr.outputs.Label(num_top_classes=3)
    iface = gr.Interface(
        fn           = predictions,
        inputs       = gr.inputs.Image(shape=(224,224)),
        outputs      = [label, 'plot', 'plot'],
        examples     = None,
        title        = 'Skin Cancer Classification',
        description  = f'Classifies image in the seven categories: {categories}',
        flagging_dir = 'output/gradio/flagged'
    ).launch(share=False, debug=True)
