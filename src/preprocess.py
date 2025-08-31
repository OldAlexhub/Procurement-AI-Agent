from httpx import head
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


#Load .ENV
load_dotenv()

#Load params from params.yaml
root = os.path.dirname(os.path.dirname(__file__)) 
params_path = os.path.join(root, "params.yaml")
with open(params_path, "r") as f:
    config= yaml.safe_load(f)["preprocess"]

#Load data
def main():

    mongo_uri = os.getenv(config['mongo_uri_env'])
    client = MongoClient(mongo_uri)
    db = client[config['db']]
    src_collection = db[config['src_collection']]
    dst_collection = db[config['dst_collection']]
    data = pd.DataFrame(list(src_collection.find()))
    # Remove mongodb's _id column
    data = data.drop(columns='_id')
    #Fill NAs with 0
    data= data.fillna(0)
    # Transform date values to three seperate columns
    data['Order_Date'] = pd.to_datetime(data['Order_Date'])
    data['Oder_Day'] = pd.to_datetime(data['Order_Date']).dt.day
    data['Oder_Month'] = pd.to_datetime(data['Order_Date']).dt.month
    data['Oder_Year'] = pd.to_datetime(data['Order_Date']).dt.year

    le = LabelEncoder()
    for col in ['Supplier', 'Item_Category', 'Order_Status','Compliance']:
        data[f"encoded_{col}"] = le.fit_transform(data[col])

    #save label encoder artifacts
    scaler_path = Path(root) / config['scaler_path']
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, scaler_path)

    #create the processed dataframe
    processed = data.select_dtypes('number')
    #Reorganize the dataframe
    processed = processed[['encoded_Supplier','encoded_Item_Category', 'encoded_Order_Status', 'Oder_Day', 
                           'Oder_Month', 'Oder_Year', 'Quantity', 'Unit_Price', 'Negotiated_Price', 'Defective_Units', 'encoded_Compliance']]

    #Save a parquet copy
    parquet_path = Path(root) / config['output_parquet']
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(parquet_path, index=False)

    if config.get("clear_dst_first"):
        dst_collection.delete_many({})

    processed= processed.to_dict(orient="records")

    dst_collection.insert_many(processed)

    print(f"Saved parquet to {parquet_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Inserted docs into {dst_collection.full_name}")

if __name__ == "__main__":
    main()