## Brain Tumour Detection and Segmentation Application

This repo contains the source code for my brain tumour detection and segmentation application. 
An emsemble model was used for classification of the magnetic resonance (MR) images, and a ResUNET model was used for segmentation. The models were deplpoyed on a web app using Streamlit.

Data preprocessing and visualization were done in [Project](https://github.com/Oguama77/Brain-Tumour-Detection-and-Segmentation-Device/blob/main/Project.ipynb), followed by model building and training in [Ensemble_model](https://github.com/Oguama77/Brain-Tumour-Detection-and-Segmentation-Device/blob/main/Ensemble_model.ipynb). The segmentation model was trained in [Segmentation_model](https://github.com/Oguama77/Brain-Tumour-Detection-and-Segmentation-Device/blob/main/Segmentation_model.ipynb), and [Model_combo](https://github.com/Oguama77/Brain-Tumour-Detection-and-Segmentation-Device/blob/main/Model_combo.ipynb) contains code to test the performance of various model combinations. 
[Model_conversion](https://github.com/Oguama77/Brain-Tumour-Detection-and-Segmentation-Device/blob/main/Model_conversion.ipynb) contains code to convert the tensorflow models to tensorflow lite models for ease of deployment.
