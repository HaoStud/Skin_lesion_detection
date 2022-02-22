# Skin lesion detection
Group Project for `Deep Learning for Medical Imaging WS21/22`
- [HAM10000](https://doi.org/10.7910/DVN/DBW86T)

## Data
1. Get data from [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)  and unzip in `./input/orig/`.
2. Run `./data/dataset.py` to create organized data structure
3. Run `./data/preprocess.py` to create augmented dataset

## Checkpoints

The checkpoints can be found [here](https://drive.google.com/drive/folders/1psbJpj-RN34UwuY0EAHrdFaCuFkfaBYZ?usp=sharing).
You can unzip them in `./output/checkpoints/`.

## Creating the environment
### Conda
```sh
conda create --name dlmb python=3.9  
conda activate dlmb  
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
```

### Pipenv
```sh
pipenv install
pipenv run \
    pip3 install --pre torch torchvision \
    -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
pipenv run pip install pytorch-lightning
pipenv shell
```
