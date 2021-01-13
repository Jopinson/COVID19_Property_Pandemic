#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from path import Path

import os,sys
parentdir = Path(os.path.abspath("../.."))
sys.path.insert(0,parentdir)

# Get the DB password
from config import db_password
from sqlalchemy import create_engine


# In[2]:


# Read the raw us cities data file
file_path = Path("../../data/raw/uscities.csv")
city_df = pd.read_csv(file_path)
print(city_df.shape)
city_df.head()


# In[3]:


# Drop the columns not required
city_df.drop(columns=["city_ascii","county_fips","lat","lng","population","density","source","military","incorporated","timezone","ranking","zips","id"],inplace=True)

print(city_df.shape)
city_df.sample(n=10)


# In[4]:


# Get cities and counties only for FL and CA
city_df_ca_fl = city_df.loc[city_df["state_id"].isin(['FL','CA'])]
city_df_ca_fl


# In[5]:


#Export the data to CSV
file_path_export=Path("../../data/processed/city_county_mapping.csv")
city_df_ca_fl.to_csv(file_path_export,index=False)


# In[6]:


#Load data to database
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic"
engine = create_engine(db_string)
city_df_ca_fl.to_sql(name='city_county_mapping', con=engine, if_exists='replace')

