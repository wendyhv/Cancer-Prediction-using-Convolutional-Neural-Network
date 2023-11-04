# Uterine-Cancer-Prediction
Uterine Cancer Prediction using CNN
# Overview
# Import Libraries
To run the code, open the Cancer_Prediction.ipynb file.
## **Import libraries**

#install dicon2jpg
!pip install dicom2jpg #used to convert DICOM images into JPG format

import dicom2jpg

# Import library for splitting folders
!pip install split-folders
import splitfolders

# Import main libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from matplotlib.image import imread 
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Import libraries for image preprocessing
from skimage import exposure

# Import libraries for data augmentation and splitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Import libraries for CNN model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Visualizing a Keras model's architecture
!pip install pydot graphviz
from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg


# EDA
# Model Architecture
# Model Performance
# Acknowledgements
[1] Fu, M., Hu, Y., Lan, T., Guan, K.-L., Luo, T. and Luo, M. (2022). The Hippo signalling pathway and its implications in human health and diseases. Signal Transduction and Targeted Therapy, [online] 7(1), pp.1–20. doi:https://doi.org/10.1038/s41392-022-01191-9.

[2] Kim N.G., Koh E., Chen X., Gumbiner B.M. E-cadherin mediates contact inhibition of proliferation through Hippo signaling-pathway components. Proc. Natl. Acad. Sci. USA. 2011;108:11930–11935. doi: 10.1073/pnas.1103345108.

[3] Romero-Perez L, Garcia-Sanz P, Mota A, Leskela S, Hergueta-Redondo M, Diaz-Martin J, Lopez-Garcia MA, Castilla MA, Martinez-Ramirez A, Soslow RA, Matias-Guiu X, Moreno-Bueno G, Palacios J. A role for the transducer of the Hippo pathway, TAZ, in the development of aggressive types of endometrial cancer. Mod Pathol. 2015;28:1492–1503.

[4] Xiao, Y. and Dong, J. (2021). The Hippo Signaling Pathway in Cancer: A Cell Cycle Perspective. Cancers, 13(24), p.6214. doi:https://doi.org/10.3390/cancers13246214.
