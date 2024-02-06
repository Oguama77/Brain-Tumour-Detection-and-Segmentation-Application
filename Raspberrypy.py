#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip3 install scikit-image


# In[2]:


import cv2
import json
import scipy.stats 
import numpy as np
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tflite_runtime.interpreter as tflite
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import dash
import dash_canvas
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from dash_canvas.utils.parse_json import parse_jsonstring
from dash_canvas.utils.image_processing_utils import segmentation_generic
from dash_canvas.utils.plot_utils import image_with_contour
from dash_canvas.utils.io_utils import image_string_to_PILImage
from dash_canvas.components import image_upload_zone
import plotly.express as px


# In[3]:


NasNet = tflite.Interpreter(model_path='NasNetModel.tflite')
NasNet.allocate_tensors()

MobileNet = tflite.Interpreter(model_path='MobileNetModel.tflite')
MobileNet.allocate_tensors()

SegmentModel = tflite.Interpreter(model_path='SegModel.tflite')
SegmentModel.allocate_tensors()

TrainedModel = [NasNet,MobileNet]


# In[4]:


# Image to segment and shape parameters
#filename = 'https://upload.wikimedia.org/wikipedia/commons/e/e4/Mitochondria%2C_mammalian_lung_-_TEM_%282%29.jpg'
filename = 'test.jpg'
img = io.imread(filename)
height, width, depth = img.shape
print(img.shape)
canvas_width = 500
canvas_height = 500
scale = canvas_width / width


# ------------------ App definition ---------------------

def title():
    return "Supervized segmentation"


def description():
    return "Segmentation of objects from annotations"

def process(image):
    # img = io.imread(filename)
    mask, has_mask = [], []
    image = cv2.resize(image, (224,224))
    labels = []
    img =  image*1./255.
    #reshaping
    # converting img into array
    img = np.array(img, dtype=np.float32)
    print(img.shape, image.shape)
    #reshaping the image from 256,256,3 to 1,256,256,3
    img = np.reshape(img, (1,224,224,3))
    for model in TrainedModel:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        output = model.get_tensor(output_details[0]['index'])
        predicts = np.argmax(output, axis=1)
        labels.append(predicts)
        is_defect = scipy.stats.mode(labels,axis=0)
    if np.argmax(is_defect)==0:
        has_mask.append(0)
        fig = px.imshow(image)
        fig.add_annotation(text = "The selected image has no tumor")
        return fig
    else:
        #Creating a empty array of shape 1,224,224,1
        X = np.empty((1,224,224,3), dtype=np.float32)
        # read the image
        img2 = image.copy()
        #resizing the image and coverting them to array of type float64
        img2 = cv2.resize(img2, (224,224))
        img2 = np.array(img2, dtype=np.float32)

        # standardising the image
        img2 -= img2.mean()
        img2 /= img2.std()
        #converting the shape of image from 224,224,3 to 1,224,224,3
        X[0,] = img2

        #make prediction of mask
        input_details = SegmentModel.get_input_details()
        output_details = SegmentModel.get_output_details()
        SegmentModel.set_tensor(input_details[0]['index'], X)
        SegmentModel.invoke()
        predict = SegmentModel.get_tensor(output_details[0]['index'])
        # if sum of predicted mask is 0 then there is not tumour
        if predict.round().astype(int).sum()==0:
            has_mask.append(0)
            mask.append('No mask :)')
            fig = px.imshow(image)
            fig.add_annotation(text = "The selected image has no tumor")
            return fig
        else:
        #if the sum of pixel values are more than 0, then there is tumour
            has_mask.append(1)
            mask.append(predict)
        pred = np.array(mask[0]).squeeze().round()
        #overlay predicted mask and MRI
        img_ = image
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = cv2.resize(img_,(224,224))
        #210,105,30 0,224,150
        img_[pred==1] = (210,105,30)
    print("ficking done")
    return px.imshow( img_)


# In[ ]:


app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.config.suppress_callback_exceptions = False

colors = {
    'background': '#ffffff',
    'background2': '#0D022E',
    'text': '#7FDBFF#4A0EFA'
}
app.layout = html.Div(style={'backgroundColor': colors['background'],'background-size': '1350px 1000px'}, 
                      children=[
        html.H1(children='MRI BRAIN TUMOR DETECTION AND SEGMENTATION TOOL',style={
        'textAlign': 'center',
        'color': '#111111'
    }),
            dcc.Markdown('''
                ## `Choose an Image you want to classify and segment, the segmentation is done automatically`.
            ''',style={
            'textAlign': 'center',
            'color': '#111111'}),
        html.Div([
            html.Div([
                html.H2(children='Test Image',style={
            'textAlign': 'center',
            'color': '#111111'}),
                dcc.Graph(
                    id='canvas',
    #                 goButtonTitle='Segmentation'
            )], className="six columns"),
            html.Div([
                html.H2(children='Segmentation Result',style={
            'textAlign': 'center',
            'color': '#111111'}),
                dcc.Graph(
                    id='segmentation',
                )
                ], className="six columns"),
            ], className="row"),
        image_upload_zone('upload-image'),
    ], className="columns")

# ----------------------- Callbacks -----------------------------


@app.callback(Output('canvas', 'figure'),
            [Input('upload-image', 'contents')])
def update_canvas_upload(image_string):
    if image_string is None:
        raise PreventUpdate
    if image_string is not None:
        img = image_string_to_PILImage(image_string)
        return px.imshow(img)

@app.callback(Output('segmentation', 'figure'),
            [Input('upload-image', 'contents')])
def update_canvas_upload(image):
    if image is None:
        fig = px.line()
        fig.add_annotation(text = "Error: Please upload an image.")
        return fig
    else:
        im = image_string_to_PILImage(image)
        
        im = np.asarray(im)
        return process(im)

if __name__ == '__main__':
    app.run_server(port=8059)

