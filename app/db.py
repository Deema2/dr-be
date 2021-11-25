import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from marshmallow import Schema, fields, pre_load, validate
from urllib.parse import quote


SERVER= "127.0.0.1"
DATABASE= "KKESH"
UID= "kkeshu"
PASS= "1234"


engine = create_engine("mssql+pymssql://{}:{}@{}:1433/{}".format(UID,(PASS),SERVER,DATABASE))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base(engine) 
# ma = Marshmallow()
