# Uterine-Cancer-Prediction
Uterine Cancer Prediction using CNN
# Overview
# Import Libraries
To run the code, open the Cancer_Prediction.ipynb file.
```
# Import necessary libraries
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
from skimage import io

# Install pydicom package, which is used for handling DICOM files
!pip install pydicom
import pydicom
import glob  # Used for file locations
from PIL import Image  # Used for image processing
from skimage.transform import resize  # Used for resizing images
import copy  # Used for creating copies of objects
import matplotlib.pyplot as plt  # Used for plotting

# Set IPython to display all outputs from a cell (not just the last one)
from IPython.core.interactiveshell import InteractiveShell

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from scipy.stats import chi2_contingency # Used for statistical testing in a contingency table
```
# Data Sources
The Cancer Genome Atlas Uterine Corpus Endometrial Carcinoma (TCGA-UCEC) data collection forms part of a broader initiative focused on fostering a research community dedicated to understanding the connection between cancer characteristics and genetic makeup.  This goal is achieved by providing clinical images paired with subjects from The Cancer Genome Atlas (TCGA).  Clinical, genetic, and pathological data are stored on the Genomic Data Commons (GDC) Data Portal, while radiological data is specifically archived in The Cancer Imaging Archive (TCIA).  Accessing and downloading the inventory file containing all UCEC data from The Cancer Genome Atlas can be further facilitated through the TCIA radiology portal.
# EDA
Random samples of gene mutations and non-genes in TCGA-UCEC
<img width="1520" alt="Screenshot 2023-11-03 at 7 37 10 PM" src="https://github.com/wendyhv/Cancer-Prediction-using-Convolutional-Neural-Network/assets/149440642/7e8302b2-d96a-44cd-b6f7-f86d35c35ff4">
# Model Architecture
The Convolutional Neural Network (CNN) constructed for the uterine cancer prediction is a sequence of layers tailored to process two-dimensional image data. The model starts with a Conv2D layer with 32 filters of size 3x3, which is the first stage of feature extraction; filters slide over the image and output a feature map that highlights features like edges and corners. Activation for this layer is 'relu', which introduces non-linearity, allowing the network to learn more complex patterns. Following this, a MaxPooling2D layer reduces the spatial dimensions of the feature maps by taking the maximum value over a 2x2 window, effectively summarizing the features detected in the prior layer while reducing computation for subsequent layers. This pattern repeats with increasing filter counts (64 and then 128 in subsequent Conv2D layers), each followed by a MaxPooling2D layer. This hierarchical structure allows the network to first learn low-level features and gradually combine them to recognize higher-level features as the data flows through the network.

Once the convolutional base has processed the images, the model's architecture flattens the three-dimensional feature maps into a one-dimensional vector. This flattened data is then fed through a Dense layer, which is a fully connected neural network layer with 64 units and 'relu' activation. This part of the network integrates the features extracted by the convolutional base to begin making classification decisions. The final Dense layer has  2 units with a 'softmax' activation function that outputs a probability distribution over two classes, which seems to be a consistency for a binary classification task, otherwise may require adjustment to match the number of target classes accurately. The 'softmax' function ensures that the output values are between 0 and 1 and sum up to 1, making it suitable for multi-class classification problems. The network is then compiled with the 'adam' optimizer, which is an adaptive learning rate method, and uses 'sparse_categorical_crossentropy' for loss calculation, which is appropriate for multi-class classification where the classes are mutually exclusive.
# Model Performance
# Acknowledgements
[1] Fu, M., Hu, Y., Lan, T., Guan, K.-L., Luo, T. and Luo, M. (2022). The Hippo signalling pathway and its implications in human health and diseases. Signal Transduction and Targeted Therapy, [online] 7(1), pp.1–20. doi:https://doi.org/10.1038/s41392-022-01191-9.

[2] Kim N.G., Koh E., Chen X., Gumbiner B.M. E-cadherin mediates contact inhibition of proliferation through Hippo signaling-pathway components. Proc. Natl. Acad. Sci. USA. 2011;108:11930–11935. doi: 10.1073/pnas.1103345108.

[3] Romero-Perez L, Garcia-Sanz P, Mota A, Leskela S, Hergueta-Redondo M, Diaz-Martin J, Lopez-Garcia MA, Castilla MA, Martinez-Ramirez A, Soslow RA, Matias-Guiu X, Moreno-Bueno G, Palacios J. A role for the transducer of the Hippo pathway, TAZ, in the development of aggressive types of endometrial cancer. Mod Pathol. 2015;28:1492–1503.

[4] Xiao, Y. and Dong, J. (2021). The Hippo Signaling Pathway in Cancer: A Cell Cycle Perspective. Cancers, 13(24), p.6214. doi:https://doi.org/10.3390/cancers13246214.
