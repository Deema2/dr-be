import keras
import os
from pathlib import Path
from config import Settings

settings = Settings()


parent_dir = Path(__file__).parents[0]
# print(parent_dir)

model_name = settings.MODEL_NAME
print("Model name: ", model_name)
model = keras.models.load_model(os.path.join(parent_dir, model_name))
# model = keras.models.load_model(os.path.join(parent_dir, 'resnet_model'))