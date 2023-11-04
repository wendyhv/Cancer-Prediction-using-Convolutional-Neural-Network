# Uterine-Cancer-Prediction
Uterine Cancer Prediction using CNN
# Overview
Uterine Corpus Endometrial Carcinoma, also commonly referred to as endometrial cancer, is one of the most common malignant tumors of the female reproductive system, occurring in the endometrium of the uterus. Timely detection and intervention play a crucial role in enhancing the prognosis and outcomes for individuals afflicted with endometrial carcinoma. The Hippo pathway is a highly sophisticated mechanism that has evolved across various species to maintain the balance of tissues by overseeing processes like cell growth, specialization, and programmed cell death. Its key components, including LATS1, LATS2, MST1, YAP1, and TAZ, play a key role in upholding the structural integrity of conventional tissue and cell equilibrium. When the Hippo signaling pathway has errors, it becomes a contributing factor to the progression and spread of tumors during cancer development. Based on this, the role of Hippo pathway signaling element gene mutation in predicting endometrial cancer is expected to be greatly improved.

The project aims to develop a convolutional neural network (CNN) machine learning model to figure out if the presence of a gene mutation will help us in detecting that cancer. Investigating how genetic mutations influence the control of key components of the Hippo pathway, such as LATS1, LATS2, MST1, YAP1, and TAZ. The whole process includes importing libraries, data sources, exploratory data analysis (EDA), model architecture, model performance evaluation, and Chi-square. This project holds the promise of improving cancer detection techniques and enhancing treatment strategies, ultimately leading to better outcomes for patients. The classifier is identifying gene mutations mixed with non-genetic elements.
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
The Cancer Genome Atlas Uterine Corpus Endometrial Carcinoma (TCGA-UCEC) data collection forms part of a broader initiative focused on fostering a research community dedicated to understanding the connection between cancer characteristics and genetic makeup. This goal is achieved by providing clinical images paired with subjects from The Cancer Genome Atlas (TCGA). Clinical, genetic, and pathological data are stored on the Genomic Data Commons (GDC) Data Portal, while radiological data is specifically archived in The Cancer Imaging Archive (TCIA). Accessing and downloading the inventory file containing all UCEC data from The Cancer Genome Atlas can be further facilitated through the TCIA radiology portal.
# EDA
Random samples of gene mutations and non-genes in TCGA-UCEC
<img width="1520" alt="Screenshot 2023-11-03 at 7 37 10 PM" src="https://github.com/wendyhv/Cancer-Prediction-using-Convolutional-Neural-Network/assets/149440642/7e8302b2-d96a-44cd-b6f7-f86d35c35ff4">
# Model Architecture
The Convolutional Neural Network (CNN) constructed for the uterine cancer prediction is a sequence of layers tailored to process two-dimensional image data. The model starts with a Conv2D layer with 32 filters of size 3x3, which is the first stage of feature extraction; filters slide over the image and output a feature map that highlights features like edges and corners. Activation for this layer is 'relu', which introduces non-linearity, allowing the network to learn more complex patterns. Following this, a MaxPooling2D layer reduces the spatial dimensions of the feature maps by taking the maximum value over a 2x2 window, effectively summarizing the features detected in the prior layer while reducing computation for subsequent layers. This pattern repeats with increasing filter counts (64 and then 128 in subsequent Conv2D layers), each followed by a MaxPooling2D layer. This hierarchical structure allows the network to first learn low-level features and gradually combine them to recognize higher-level features as the data flows through the network.

Once the convolutional base has processed the images, the model's architecture flattens the three-dimensional feature maps into a one-dimensional vector. This flattened data is then fed through a Dense layer, which is a fully connected neural network layer with 64 units and 'relu' activation. This part of the network integrates the features extracted by the convolutional base to begin making classification decisions. The final Dense layer has  2 units with a 'softmax' activation function that outputs a probability distribution over two classes, which seems to be a consistency for a binary classification task, otherwise may require adjustment to match the number of target classes accurately. The 'softmax' function ensures that the output values are between 0 and 1 and sum up to 1, making it suitable for multi-class classification problems. The network is then compiled with the 'adam' optimizer, which is an adaptive learning rate method, and uses 'sparse_categorical_crossentropy' for loss calculation, which is appropriate for multi-class classification where the classes are mutually exclusive.
# Model Performance
(1, 0): 0.9577 -  prediction of LATS1 presence when Uterine cancer

(2, 0): 0.9366 -  prediction of LATS2 presence when Uterine cancer

(3, 0): 0.8085 -  prediction of MST1 presence when Uterine cancer

(4, 0): 0.8085 -  prediction of YAP1 presence when Uterine cancer

(5, 0): 0.8214 -  prediction of TAZ presence when Uterine cancer
<img width="1060" alt="Screenshot 2023-11-03 at 9 16 50 PM" src="https://github.com/wendyhv/Cancer-Prediction-using-Convolutional-Neural-Network/assets/149440642/a79ddb9f-fd6c-4618-9257-104e000196d5">
These accuracy results indicate the performance of separate models or the same model tested with different datasets for the prediction of the presence of various genes when Uterine cancer is present. With an accuracy of 95.77%, the model is quite reliable in predicting the presence of the LATS1 gene in Uterine cancer cases. The accuracy drops slightly to 93.66% for the LATS2 gene, but it is still a high level of accuracy, indicating the model's effectiveness. The accuracy further drops to 80.85% for the MST1 gene. While this is still above average, it is noticeably lower than the accuracy for LATS1 and LATS2, suggesting that predicting MST1 might be more challenging or that the model may require further tuning for this gene. YAP1 also has an accuracy of 80.85%, indicating similar challenges as with MST1. It could be due to the gene's variability in expression or other factors that make it less distinguishable by the model. Finally, TAZ's presence has an accuracy of 82.14%, a slight improvement over MST1 and YAP1 but still below the performance for LATS1 and LATS2.

This model has strong test accuracy, especially applicable to LATS1 and LATS2. LATS1/LATS2 is strongly associated with uterine cancer. The results showed that the model was more effective in predicting LATS1 and LATS2 gene mutations in patients with uterine cancer. We then repeated this experiment in different types of cancer and found that LATS1 and LATS2 were equally high in detection power or accuracy in other cancers, meaning that LATS1 and LATS2 gene mutations are important, but they may not be the most effective specific gene mutations to detect TCGA-UCEC cancers.

Next we flipped the model to test whether a given gene mutation could help us predict cancer, and we used the model to distinguish between uterine and non-uterine cancers based on LATS1 mutations. Then we choose two case ids. One is all uterine cancer LATS1 case id, labeled 1, and the second is non-uterine cancer LATS1 case id, which may be two or three other cancers. They are labeled 0. The accuracy of the test results showed that the accuracy of the model to predict whether the LATS1 gene mutation was in uterine cancer patients was 85.86%, which had high reliability. This high degree of accuracy shows that our model can effectively help predict the presence of uterine cancer based on specific genetic mutations.
# Chi-Square
Since the prediction accuracy of LATS1 and LATS2 is very high and the results are very close, in order to determine the correlation between the LATS1 and LATS2 genes, we decided to perform chi-square tests on the LATS1 and LATS2 test sets. The higher the Chi-square value, the greater the degree of deviation. The chi-square value of 280.01 and a very low p-value of approximately 7.46e-63 indicate that there is a significant difference between the distributions of categorical data in X1_test and X2_test. In other words, the images in these two sets exhibit significantly different patterns of pixel values.

The actual Chi-square statistic also tells us the effect size or the strength of the association. Although chi-square tests do not directly tell us the nature of the difference or the strength of the relationship, a high chi-square value indicates a potentially strong association. In a medical context, further evaluation is needed to determine whether these differences are meaningful in terms of diagnosis, prognosis, or treatment.
# Acknowledgements
[1] Fu, M., Hu, Y., Lan, T., Guan, K.-L., Luo, T. and Luo, M. (2022). The Hippo signalling pathway and its implications in human health and diseases. Signal Transduction and Targeted Therapy, [online] 7(1), pp.1–20. doi:https://doi.org/10.1038/s41392-022-01191-9.

[2] Kim N.G., Koh E., Chen X., Gumbiner B.M. E-cadherin mediates contact inhibition of proliferation through Hippo signaling-pathway components. Proc. Natl. Acad. Sci. USA. 2011;108:11930–11935. doi: 10.1073/pnas.1103345108.

[3] Romero-Perez L, Garcia-Sanz P, Mota A, Leskela S, Hergueta-Redondo M, Diaz-Martin J, Lopez-Garcia MA, Castilla MA, Martinez-Ramirez A, Soslow RA, Matias-Guiu X, Moreno-Bueno G, Palacios J. A role for the transducer of the Hippo pathway, TAZ, in the development of aggressive types of endometrial cancer. Mod Pathol. 2015;28:1492–1503.

[4] Xiao, Y. and Dong, J. (2021). The Hippo Signaling Pathway in Cancer: A Cell Cycle Perspective. Cancers, 13(24), p.6214. doi:https://doi.org/10.3390/cancers13246214.
