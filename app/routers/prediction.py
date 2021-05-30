from fastapi import Depends, FastAPI, HTTPException, File, UploadFile, APIRouter
import pandas as pd
import numpy as np
import io 
import os
import sys
import uvicorn
import operator
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,array_to_img, load_img,save_img
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
pd.set_option('display.max_columns', None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml_model.ML_Model import model
import cv2
import PIL
import io
IMG_SIZE = 224
IMG_PREPROCESS_SIZE = 512

router = APIRouter()


def GUASS_BLUR(img):
    img = cv2.resize(img, (IMG_PREPROCESS_SIZE,IMG_PREPROCESS_SIZE))
    img_t = cv2.addWeighted(img,4, cv2.GaussianBlur(img , (0,0) , 22) ,-4 ,128)
    img_t = cv2.resize(img_t, (IMG_SIZE,IMG_SIZE))
    return img_t


def CROP_ROI(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # Find contour and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        break
    ROI = GUASS_BLUR(ROI)
    return ROI


def preprocess_image(file):
    img = CROP_ROI(file) 
    print(type(img))
    print(img.shape)
    # cv2.imwrite("img1.png", cv2.resize(img, (IMG_PREPROCESS_SIZE,IMG_PREPROCESS_SIZE)))
    
    img = np.expand_dims(img, axis=0)
    # cv2.imwrite("expimg.png",img)
    return img

@router.post('/predict')
async def upload_file(retina_img: UploadFile = File(...), target: str = None):   
    # 1- Read image
    contents = await retina_img.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('RGB')
    open_cv_image = np.array(pil_image) 
    
    # 2- Convert RGB image to BGR image
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    # 3- Preprocess image:
    img = preprocess_image(open_cv_image)
    
    # 4- Predict label if image:
    prediction = model.predict(img)
    prediction = list(prediction[0])
    print(prediction)
    max_pred = max(prediction)
    max_pred_index = prediction.index(max_pred)
    max_pred = round(max_pred, 2)
    return {"Predicted class is: " + str(max_pred_index) + " with a probability of: " + (str(max_pred*100)) + "%"}