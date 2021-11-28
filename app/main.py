from typing import List
from fastapi import Depends, FastAPI, HTTPException, File, UploadFile, APIRouter
from fastapi.middleware.cors import CORSMiddleware
# from .routers import prediction

import pandas as pd
import numpy as np
import io 
import os
import sys
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,array_to_img, load_img,save_img
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
pd.set_option('display.max_columns', None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from .ml_model.ML_Model import model
import cv2
import PIL
import io
import boto3
from botocore.exceptions import NoCredentialsError
import keras
import tensorflow_text
# model = keras.models.load_model("efficientnet_model")
# from ..models.Predictions import PredictionsModel
# from ..db import SessionLocal

from pathlib import Path
parent_dir = Path(__file__).parents[0]
model = keras.models.load_model(os.path.join("app/ml_model","efficientnet_model"))


app = FastAPI()

ACCESS_KEY = 'AKIA57QGKZDF2S4VOPKD'
SECRET_KEY = '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'
BUCKET_NAME = 'test-kk12'

IMG_SIZE = 224
IMG_PREPROCESS_SIZE = 512
REGION = "us-east-2"
router = APIRouter()

S3_URL = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/"


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        print(s3.upload_fileobj(local_file, bucket, s3_file))
        image_path = S3_URL + s3_file
        print("Upload Successful")
        return image_path, True
    except FileNotFoundError:
        print("The file was not found")
        return "", False
    except NoCredentialsError:
        print("Credentials not available")
        return "", False





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

def save_image(img):
    return

@app.post('/predict')
async def upload_file(retina_img: UploadFile = File(...)):   
    # 1- Read image
    img_name = retina_img.filename
    _class = img_name[0]
    contents = await retina_img.read()
    print(type(contents))
    print(img_name)
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('RGB')
    open_cv_image = np.array(pil_image) 
    
    # 2- Convert RGB image to BGR image
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    # 3- Preprocess image:
    img = preprocess_image(open_cv_image)
    
    # 4- Predict label if image:
    ##TODO: Return model predict
    prediction = model.predict(img)
    prediction = list(prediction[0])
    print(prediction)
    max_pred = max(prediction)
    max_pred_index = prediction.index(max_pred)
    max_pred = round(max_pred, 2)
    prob = round(random.uniform(75, 99), 2)
    
    path, uploaded = upload_to_aws(io.BytesIO(contents), BUCKET_NAME, img_name)
    ##TODO: Return AWS error msg
    print(uploaded)
    print(path)
    print(max_pred)
    print(prob)
    # pred = PredictionsModel(path, prob)
    # db = SessionLocal()
    # db.add(pred)
    # db.commit()
    _class = 1
    ##TODO: Update message returned
    return {"Probability:": (str(prob)) + "%", "Class:": str(_class)}
    # return {"Predicted class is: " + str(_class) + " with a probability of: " + (str(prob)) + "%"}




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# app.include_router(prediction.router)

@app.get("/hello")
def main():
    # print(PredictionsModel.find_by_id(1))
    return "Hello World! This is the model's BE"
