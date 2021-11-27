from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    DEBUG: bool = False
    PORT: int = 5600
    basedir = os.path.abspath(os.path.dirname(__file__))

    """
    DB Config:
    """
    SQLALCHEMY_ECHO: bool =  False
    SQLALCHEMY_TRACK_MODIFICATIONS: bool =  True
    SQLALCHEMY_DB_URL: str = os.getenv("DB_URL", "")

    """
    ML Model:
    """
    MODEL_NAME: str = os.getenv("MODEL_NAME", "efficientnet_model")
    # MODEL_NAME: str = os.getenv("MODEL_NAME", "resnet_model")

