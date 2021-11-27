from typing import List
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routers import prediction
app = FastAPI()
from db import engine
from .models.Predictions import PredictionsModel

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(prediction.router)

@app.get("/hello")
def main():
    print(PredictionsModel.find_by_id(1))
    return "Hello World! This is the model's BE"
