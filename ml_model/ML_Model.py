import keras
import os
from pathlib import Path

parent_dir = Path(__file__).parents[0]
print(parent_dir)
model = keras.models.load_model(os.path.join(parent_dir, 'resnet_model'))
