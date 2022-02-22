#!/bin/env pipenv-shebang

from model.models import train

if __name__ == '__main__':
    model_names = [
        'resnet-drop',
        # 'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
    ]

    for name in model_names:
        train(model_name=name)
