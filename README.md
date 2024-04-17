# Boosting FSMIS with High-Confidence Prior Mask


### Abstract
Labeling large amounts of medical data is travailing, leading to the blooming of few-shot medical image segmentation, which aims to segment the foreground of a query image given a labeled support set. Almost all current models adopt the cosine distance to measure the similarity between prototypes and query features. However, the limitation of the cosine distance is exacerbated by intra-class differences and inter-class imbalances in medical image scenarios, where angle-only evaluation can induce misclassification to under- and over-segmentation. Motivated by this, we propose a High-Confidence Prior Mask-guided Network (HCPMNet), comprising a High-Confidence Mask Generator (HCPMG), a Target Region Mining (TRM) module, and a Prototype-Oriented Expansion Match (POEM) module. Our HCPMNet offers key advantages: 1) HCPMG is the first to combinatively evaluate angle and magnitude similarity, generating high-confidence priori masks that accurately and completely localize target regions. 2) TRM mines and aggregates target class information under the guidance of priori masks. 3) POEM, based on both similarity metrics, correctly matches prototypes with query features. Extensive experiments on three general medical datasets show that our HCPMNet achieves a new SoTA with great superiority (Average improvement up to 3.11\%).

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 
