# Uterine-Cancer-Prediction
Uterine Cancer Prediction using CNN
# Overview
# Import Libraries
To run the code, open the Cancer_Prediction.ipynb file.
'# Import necessary libraries
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
import matplotlib.pyplot as plt  # Used for plotting'

# EDA
# Model Architecture
# Model Performance
# Acknowledgements
[1] Fu, M., Hu, Y., Lan, T., Guan, K.-L., Luo, T. and Luo, M. (2022). The Hippo signalling pathway and its implications in human health and diseases. Signal Transduction and Targeted Therapy, [online] 7(1), pp.1–20. doi:https://doi.org/10.1038/s41392-022-01191-9.

[2] Kim N.G., Koh E., Chen X., Gumbiner B.M. E-cadherin mediates contact inhibition of proliferation through Hippo signaling-pathway components. Proc. Natl. Acad. Sci. USA. 2011;108:11930–11935. doi: 10.1073/pnas.1103345108.

[3] Romero-Perez L, Garcia-Sanz P, Mota A, Leskela S, Hergueta-Redondo M, Diaz-Martin J, Lopez-Garcia MA, Castilla MA, Martinez-Ramirez A, Soslow RA, Matias-Guiu X, Moreno-Bueno G, Palacios J. A role for the transducer of the Hippo pathway, TAZ, in the development of aggressive types of endometrial cancer. Mod Pathol. 2015;28:1492–1503.

[4] Xiao, Y. and Dong, J. (2021). The Hippo Signaling Pathway in Cancer: A Cell Cycle Perspective. Cancers, 13(24), p.6214. doi:https://doi.org/10.3390/cancers13246214.
