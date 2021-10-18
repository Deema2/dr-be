import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from marshmallow import Schema, fields, pre_load, validate
from urllib.parse import quote


SQLALCHEMY_DATABASE_URL = os.getenv("DB_CONN")


SERVER= "localhost"
DATABASE= "KKESH"
UID= os.getenv( "DB_USER")
PASS= os.getenv( "DB_PWD")



engine = create_engine("mssql+pymssql://{}:{}@{}:1433/{}".format(UID,quote(PASS),SERVER,DATABASE))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base(engine) 
ma = Marshmallow()
