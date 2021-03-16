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

### Mosmed-Test
Finally, download the test dataset that we released at [google disk](). Unzip archive `test.zip`. You obtain the folder
```
- .../test
    - covid
        - images
            - *.nii.gz
        - masks
            - *.nii.gz
    - bacterial_pneumonia
    - nodules
    - normal
```
To prepare this Nifti dataset for testing run
```
python scripts/prepare_test.py -i <raw_test_root> -o <test_root>
```
where `<raw_test_root> = .../test`.

## Training models

### Lungs segmentation
First we need to train lungs segmentation network. It is used in the COVID-19 triage pipelines at the preprocessing step to crop input image to lungs bounding box.

To train network run
```
python scripts/train_lungs_sgm.py --nsclc <nsclc_root> -o experiments/lungs_sgm
```
Then run 
```
python scripts/predict_dataset_lungs_masks.py --model experiments/lungs_sgm/checkpoints/best/Sequential --dataset <mosmed_root>
```
and
```
python scripts/predict_dataset_lungs_masks.py --model experiments/lungs_sgm/checkpoints/best/Sequential --dataset <test_root>
```
to predict and save lungs masks for the Mosmed-1110 and Mosmed-Test datasets.

### Proposed multitask model

To train the proposed multitask network for COVID-19 triage run

```
python scripts/train_multitask_spatial.py --mosmed <mosmed_root> --medseg <medseg_root> --nsclc <nsclc_root> -o experiments/multitask_spatial
```

### ResNet-50
To train ResNet-50 for COVID-19 identification run
```
python scripts/train_resnet.py --mosmed <mosmed_root> --nsclc <nsclc_root> -o experiments/resnet
```

## Results

### Proposed multitask model
To evaluate the trained proposed multitask model on the test dataset and save predictions to `<test_predictions_root>` run
```
python scripts/eval_multitask_spatial.py --lungs_model experiments/lungs_sgm/checkpoints/best/Sequential --mutltitask_spatial experiments/multitask_spatial/checkpoints/best/MultitaskSpatial --test <test_root> -o <multitask_spatial_predictions_root>
```
To calculate metrics run
```
python scripts/calculate_metrics.py --test <test_root> --pred <multitask_spatial_predictions_root>
```

### ResNet-50
To evaluate ResNet-50 run
```
python scripts/eval_resnet.py --lungs_model experiments/lungs_sgm/checkpoints/best/Sequential --resnet experiments/resnet/checkpoints/best/Sequential --test <test_root> -o <resnet_predictions_root>
```
To calculate metrics run
```
python scripts/calculate_metrics.py --test <test_root> --pred <resnet_predictions_root>
```
