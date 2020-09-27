import os 
import io
import datetime
import time
from flask import Flask, jsonify, request
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import pickle
import cv2
import torch
from torch.autograd import Variable
from torchvision import models
from torch import nn
from collections import OrderedDict

# Initialize the Flask application
app = Flask(__name__)

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg','jpeg','png','bmp','webp'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    resp = jsonify({'message' : 'Plant Disease Recognition API. Built by Nitish Srivastava.'})
    resp.status_code = 405
    return resp    

@app.route('/api/upload_image', methods=['GET','POST'])
def upload_image():
    if request.method == "GET":
        resp = jsonify({'message' : 'Method not allowed'})
        resp.status_code = 405
        return resp
    else:
        # check if the post request has the files part
        if 'plant_image' not in request.files:
            resp = jsonify({'message' : 'No file in the request'})
            resp.status_code = 400
            return resp  
        # Get the list of the uploaded files
        file = request.files['plant_image']

        
        if file and allowed_file(file.filename):
            if file is not None :
                start = time.time()
                
                image = Image.open(BytesIO(file.read()))
                loaded_model, class_to_idx = load_checkpoint('plants9615_checkpoint.pth')
                
                idx_to_class = { v : k for k,v in class_to_idx.items()}

                p, c = predict(image, loaded_model, idx_to_class)

                p_indx = np.argmax(p)
                p_data = p[p_indx]*100
                c_data = c[p_indx]
                
                end = time.time()
                print('Processing time : {} '.format(end-start))

                result = {'prediction': '%.2f' % (p_data), 'class': c_data}

                resp = jsonify({'message':'success', 'data':result})
                resp.status_code = 201
                return resp
            else :
                resp = jsonify({'message' : f"Error : Error loading image file"})
                resp.status_code = 405
                return resp
        else:
            resp = jsonify({'message' : f"Error : Error loading image file"})
            resp.status_code = 405
            return resp

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = models.resnet152()
    
    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 39
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 39)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage

def predict(image, model, idx_to_class, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = torch.FloatTensor([process_image(image)])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=False, processes=3)

