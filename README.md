# Towards Large-scale Single-shot Millimeter-wave Imaging for Low-cost Security Inspection
[![DOI](https://zenodo.org/badge/736483770.svg)](https://zenodo.org/doi/10.5281/zenodo.11091264)

## 1. System requirements
### 1.1 All software dependencies and operating systems
The project has been tested on Ubuntu 22.04.1 LTS.
### 1.2 Versions the software has been tested on
The project has been tested on Python version >= 3.8.
### 1.3 Any required non standard hardware
There is no non-standard hardware required for this project. 



## 2. Installation guide
### 2.1 Download the repository

To install the software, clone the repository and run the following command in the terminal:
```
git clone https://github.com/bianlab/MMW.git
```

### 2.2 Install the requirements 

To install the requirements, make sure that you have installed Anaconda, and run the following command in the terminal:
```
conda create -n mmw python=3.8
conda activate mmw

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm scipy
```

## 3. Demo
### 3.1 Untrained learning reconstruction
We have provided some test cases in the `Reconstruction/ExemplarData` folder. You can run the following command in the terminal:
```
cd ./Reconstruction
python Main.py --data_folder ExemplarData/ --data_name data_0.25 --n_iter 1000 --use_init True
```
The reconstructed 3D data (.mat) is in the `Reconstruction/Result` folder.

### 3.2 Concealed target detection
We have provided some test cases in the `Detection/images/0.1` and `Detection/images/0.25` folders corresponding to 10\% sampling ratio reconstructions and 25\% sampling ratio reconstructions.
There are two detection networks in the `Detection/models` folder, one for 10\% sampling ratio, and the other for 25\% sampling ratio. Make sure you have yolov8 package or run the command:
```
pip install ultralytics
```
You can run the following command in the terminal:
```
cd ./Detection
yolo predict imgsz=640 model=models/0.1.pt source=images/0.1 name=0.1 show_conf=False iou=0.5
yolo predict imgsz=640 model=models/0.25.pt source=images/0.25 name=0.25 show_conf=False iou=0.5
```
The detection images are in the `Detection/runs/detect/0.1`  or `Detection/runs/detect/0.25` folder.


## 4. Instructions for use
### 4.1 How to run the reconstruction code on your data
We have provided the sparse pattern (.mat) in the `Mask` folder. You can make your own input data for untrained learning by multiplying the mask by the echoes. The input data (.mat) consists of 4 parts: 

`S_echo_real_sampling`: real part of the sparsely sampled echo

`S_echo_imag_sampling`: imaginary part of the sparsely sampled echo

`sampling_mask`: sparse pattern

`scene_init`: initial reconstruction by RMA or other method (optional)

You can run the following command in the terminal:
```
python Main.py --data_folder your_folder --data_name you_own_data --n_iter iteration_number --use_init True
```
### 4.1 How to run the detection code on your data
Put your images in the `Detection/images/0.1` or `Detection/images/0.25` folders. Then run the following command in the terminal:
```
yolo predict imgsz=640 model=models/0.1.pt source=images/0.1 name=0.1 show_conf=False iou=0.5
yolo predict imgsz=640 model=models/0.25.pt source=images/0.25 name=0.25 show_conf=False iou=0.5
```
The code will check both the folders and produce the detection results.
