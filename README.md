## Brain Tumour Detection and Segmentation Application

This repo contains the source code for my brain tumour detection and segmentation web application. 
An emsemble model was used for classification of the magnetic resonance (MR) images, and a ResUNET model was used for segmentation. The models were deplpoyed on a web app using Streamlit.

Data preprocessing and visualization was done on [Project](https://github.com/Oguama77/Brain-Tumour-Detection-and-Segmentation-Device/blob/main/Project.ipynb), followed by model building and training in TheEnsembleModel. The segmentation model was trained in TheSegmentationModel, and ModelCOmbo contains code to 
test the performance of various model combinations. 
Modelconversion contains code to convert the tensorflow models to tensorflow lite models for ease of deployment.
