# CT-based COVID-19 Triage: Deep Multitask Learning Improves Joint Identification and Severity Quantification

This repository is the official implementation of [CT-based COVID-19 Triage: Deep Multitask Learning Improves Joint Identification and Severity Quantification]().

## Requirements

To install requirements:

```setup
conda create -n covid_19_triage python=3.6
conda activate covid_19_triage
pip install -e .
```

## Datasets
For training and validation of neural networks for COVID-19 triage we used several public datasets:
- [Mosmed-1110 dataset](https://mosmed.ai/)
- [Medseg-20 and Medseg-9 datasets](http://medicalsegmentation.com/covid19/)
- [NSCLC-Radiomics dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics)
We also used [lung annotations for NSCLC-Radiomics dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68551327) and [LIDC-IDRI (a.k.a. LUNA16) dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) to train lung segmentation network.

### Mosmed-1110

### Medseg-20 and Medseg-9

### NSCLC-Radiomics

### LIDC-IDRI

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
