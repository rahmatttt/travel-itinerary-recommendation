import pymysql
import pandas as pd
from sqlalchemy import create_engine
import os
import re

# default connection variables
host = "127.0.0.1"
user = "root"
passwd = ""
db = "rekomendasi_wisata"

def read_from_sql(query, host = host, user=user, passwd=passwd, db=db):
    try:
        mydb = pymysql.connect(
            host=host,  # your host, usually localhost
            user=user,  # your username
            passwd=passwd,  # your password
            db=db,
            port=3306,
        )

        dataset = pd.read_sql(query, mydb)
        mydb.close()  # close the connection
        return dataset

    except Exception as e:
        #     mydb.close()
        raise e

def execute_from_sql(query, host = host, user=user, passwd=passwd, db=db):
    try:
        mydb = pymysql.connect(
            host=host,  # your host, usually localhost
            user=user,  # your username
            passwd=passwd,  # your password
            db=db,
            port=3306,
        )
        
        cursorObject = mydb.cursor()
        cursorObject.execute(query)
        
        mydb.commit()
        mydb.close()  # close the connection
        return "success execute"

    except Exception as e:
        #     mydb.close()
        raise e

def write_df_to_sql(df, if_exists, table_name, host = host, user=user, passwd=passwd, db=db, dtype={}):
    try:
        engine = create_engine(f"mysql+pymysql://{user}:{passwd}@{host}:3306/{db}")
        if dtype == {}:
            df.to_sql(con=engine, name=table_name, if_exists=if_exists, index=False)
        else:
            df.to_sql(con=engine, name=table_name, if_exists=if_exists, index=False, dtype=dtype)
        return f"success write to {db} as {table_name}"
    except Exception as e:
        raise e
