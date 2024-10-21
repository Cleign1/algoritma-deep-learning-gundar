# Algoritma Deep Learning Gundar

This repository contains deep learning algorithms and examples using Jupyter Notebooks. untuk matakuliah algoritma deep learning

## Contents

- **Notebooks**: Jupyter Notebooks demonstrating various deep learning algorithms and their applications.
- **Datasets**: datasets used in the notebooks.
- **requirements.txt**: list of required packages for the notebooks.
- **README.md**: this file.
- **gitignore**: list of files and directories to be ignored by Git.
- **Python**: Python scripts for testing purposes.

## Requirements

- Python 3.10.15
- Tensorflow 2.17.0 With CUDA 12.4

## Datasets
- Dataset Link 10 Class : [GDrive](https://drive.google.com/file/d/1p8Flgeg-pd1pNdc4-KMO3vKlT8XLc1wt/view?usp=sharing)
- Dataset Link 3 Class : [GDrive](https://drive.google.com/file/d/1Q2P5aRAs04egYSccMXJanY44ABlJcHVf/view?usp=sharing)
- Dataset Link 2 Class : [GDrive](https://drive.google.com/file/d/17wxt5XNTo9c-VNlhzRDE8OTpUrTwRY-B/view?usp=sharing)

## Installation
1. Create a compute vm instance on GCP with the following specifications:
    - OS: Google, Deep Learning VM for TensorFlow 2.17 with CUDA 12.3, M125, Debian 11, Python 3.10, with TensorFlow 2.17 for CUDA 12.3 with Intel MKL-DNN preinstalled.
    - Machine type: n1-standard-8 (8 vCPUs, 30 GB memory)
    - GPU: NVIDIA Tesla T4
    - Boot disk: 150 GB SSD
    - Allow HTTP traffic
    - Allow HTTPS traffic
    - Insert your own SSH key

2. SSH into the VM instance and clone the repository:
    ```bash
    git clone "https://github.com/cleign1/algoritma-deep-learning-gundar.git"
    ```
3. Check for active conda environment:
    ```bash
    conda info --env
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5. SSH into the VM instance from your desired code editor and start working on the notebooks.
