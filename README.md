Dataset from:
[Kaggle](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/data)

## Setup

Create a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data-dir data/raw --output checkpoints/flower_classifier.pt
```

The script uses a ResNet-18 backbone and saves a checkpoint with the class names.

## Predict

```bash
python infer.py --checkpoint checkpoints/flower_classifier.pt --image path/to/flower.jpg
```

## Run The App

```bash
streamlit run app.py
```
