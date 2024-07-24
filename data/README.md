## Convert Script for Anomaly Detection Dataset
The convert script of the data set used in the paper.

### Usage
#### Step 1. Download the Anomaly Detection Dataset
Download and unzip the required datasets. 

| datastes     | link                               |
| ------------ | :--------------------------------- |
| MVTec AD     | https://tinyurl.com/mvtecad        |
| AITEX        | https://tinyurl.com/aitex-defect   |
| SDD          | https://tinyurl.com/KolektorSDD    |
| ELPV         | https://tinyurl.com/elpv-crack     |
| Optical      | https://tinyurl.com/optical-defect |
| Mastcam      | https://tinyurl.com/mastcam        |
| BrainMRI     | https://tinyurl.com/brainMRI-tumor |
| HeadCT       | https://tinyurl.com/headCT-tumor   |
| Hyper-Kvasir | https://tinyurl.com/hyper-kvasir   |

The normal samples in MVTec AD are split into training and test sets following the original settings. In other datasets, the normal samples are randomly split into training and test sets by a ratio of 3/1. 

- **MVTec AD**  is a popular defect inspection benchmark that has 15 different classes, with each anomaly class containing one to several subclasses. In total the dataset contains 73 defective classes of fine-grained anomaly classes at the texture- or object-level. 
- **AITEX**  is a fabrics defect inspection dataset that has 12 defect classes, with pixel-level defect annotation. We crop the original 4096 × 256 image to several 256 × 256 patch image and relabel each patch by pixel-wise annotation. 
- **SDD**  is a defect inspection dataset images of defective production items with pixel-level defect annotation. We vertically and equally divide the original 500 × 1250 image into three segment images and relabel each image by pixel-wise annotation. 
- **ELPV**  is a solar cells defect inspection dataset in electroluminescence imagery. It contains two defect classes depending on solar cells: mono- and poly-crystalline. 
- **Optical**  is a synthetic dataset for defect detection on industrial optical inspection. The artificially generated data is similar to real-world tasks. 
- **Mastcam**  is a novelty detection dataset constructed from geological image taken by a multispectral imaging system installed in Mars exploration rovers. It contains typical images and images of 11 novel geologic classes. Images including shorter wavelength (color) channel and longer wavelengths (grayscale) channel and we focus on shorter wavelength channel in this work. 
- **BrainMRI**  is a brain tumor detection dataset obtained by magnetic resonance imaging (MRI) of the brain. 
- **HeadCT**  is a brain hemorrhage detection dataset obtained by CT scan of head. 
- **Hyper-Kvasir**  is a large-scale open gastrointestinal dataset collected during real gastro- and colonoscopy procedures. It contains four main categories and 23 subcategories of gastro- and colonoscopy images. This work focuses on gastroscopy images with the anatomical landmark category as the normal samples and the pathological category as the anomalies.

#### Step 2. Running the Convert Script
Running the corresponding convert script with the argument `dataset_root` as the root of the dataset.

e.g:
```bash
python convert_AITEX.py --dataset_root=./AITEX
```
