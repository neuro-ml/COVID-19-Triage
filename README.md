# CT-based COVID-19 Triage: Deep Multitask Learning Improves Joint Identification and Severity Quantification

This repository is the official implementation of [CT-based COVID-19 Triage: Deep Multitask Learning Improves Joint Identification and Severity Quantification]().

## Requirements

To install requirements:

```
conda create -n covid_19_triage python=3.6
conda activate covid_19_triage
git clone https://github.com/neuro-ml/COVID-19-Triage
cd COVID-19-Triage
pip install -e .
```

## Datasets
### Mosmed-1110
To download the dataset run
```
python scripts/download_mosmed.py -o <raw_mosmed_root>
```
To prepare the data for training run
```
python scripts/prepare_mosmed.py -i <raw_mosmed_root> -o <mosmed_root>
```
Here `<raw_mosmed_root>`, `<mosmed_root>` are some paths under your file system, where you want to save the raw and prepared data, correspondingly.

### Medseg-20 and Medseg-9
Download [Medseg-20](http://medicalsegmentation.com/covid19/) and [Medseg-9](https://zenodo.org/record/3757476#.Xp0FhB9fgUE) datasets. Unzip archives to the folders under `<raw_medseg_root>`. You should get the following folders structure
```
- <raw_medseg_root>
    - COVID-19-CT-Seg_20cases
        - *.nii.gz
    - Infection_Mask
    - Lung_Mask
    - rp_im
    - rp_lung_msk
    - rp_msk
```
Then run
```
python scripts/prepare_medseg.py -i <raw_medseg_root> -o <medseg_root>
```

### NSCLC-Radiomics
Download the [dicoms](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) and [lungs masks](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68551327). You should get the
folders structure
```
- <raw_nsclc_root>
    - NSCLC-Radiomics
        - LUNG1-*
    - Thoracic_Cavities
        - LUNG1-*
```
Then run
```
python scripts/prepare_nsclc.py -i <raw_nsclc_root> -o <nsclc_root>
```

## Training models

### Lungs segmentation
First we need to train lungs segmentation network. It is used in the COVID-19 triage pipelines at the preprocessing step to crop input image to lungs bounding box.

To train network run
```
python scripts/train_lungs_sgm.py --nsclc <nsclc_root> -o experiments/lungs_sgm
```
Then run 
```
python scripts/predict_mosmed_lungs_masks.py --model experiments/lungs_sgm/checkpoints/best/Sequential --mosmed <mosmed_root>
```
to predict and save lungs masks for the Mosmed-1110 dataset.

## Proposed multitask model

To train the proposed multitask network run

```eval
python scripts/train_multitask_spatial.py --mosmed <mosmed_root> --medseg <medseg_root> --nsclc <nsclc_root> -o experiments/multitask_spatial
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
