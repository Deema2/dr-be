
from db import Base, SessionLocal
from sqlalchemy import PrimaryKeyConstraint

class PredictionsModel(Base):
    __tablename__ = "predictions"
    __table_args__ = (PrimaryKeyConstraint('image_path', 'prediction'), {'autoload': True,'schema':'dbo'})


    def __init__(self, image_path, prediction):
        self.image_path = image_path
        self.prediction = prediction

    @classmethod
    def find_by_id(cls, id):
        db = SessionLocal()
        return db.query(PredictionsModel).all()

