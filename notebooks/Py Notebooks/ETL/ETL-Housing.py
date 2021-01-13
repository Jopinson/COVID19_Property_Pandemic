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


# ## ETL - Extract Data

# In[2]:


# Read the raw zillow data file
file_path = Path("../../data/raw/zillow_data/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_mon.csv")
zillow_df = pd.read_csv(file_path)
zillow_df.tail()


# ## ETL - Transform Process
# Keep only FL and CA data and then drop unnecessary columns

# In[3]:


# Filter rows with state FL and CA only
state = ["FL", "CA"]
State_data = zillow_df[zillow_df.StateName.isin(state)]
State_data.dropna(axis=0,how="any",inplace=True)
State_data.tail()


# In[4]:


# Drop the unnecessary columns
State_dropped=State_data.drop(columns=["RegionID", "SizeRank", "RegionType"])
print(State_dropped.shape)
State_dropped.head()


# In[5]:


## Convert the Date columns to rows
FL_CA_df = pd.melt(State_dropped, id_vars=["RegionName","StateName"], 
                  var_name="Date", value_name="Avg_Price")


# In[6]:


FL_CA_df.set_index("RegionName")


# In[7]:


# Drop the Rows where any field is null or blank
FL_CA_df.dropna(axis=0)


# In[8]:


# Convert the date field to Datetime field
FL_CA_df["Date"] = pd.to_datetime(FL_CA_df["Date"])
FL_CA_df.head()


# In[9]:


# Remove State from the RegionName
FL_CA_df['RegionName'] = FL_CA_df.apply(lambda x:x['RegionName'][:-4], axis=1)
FL_CA_df['RegionName']


# In[10]:


# Ensure Avg_Price field is of type integer 
FL_CA_df["Avg_Price"].dtype
FL_CA_df['Avg_Price'] = FL_CA_df['Avg_Price'].astype('Int64')
FL_CA_df.tail()


# In[11]:


FL_CA_df.rename(columns={"RegionName":"region_name","StateName":"state_name","Date":"date","Avg_Price":"avg_price"},inplace=True)
FL_CA_df


# ## ETL - Load data 
# write the processed data to CSV as well as load the data to POSTGRES data base

# In[12]:


#Export the data to CSV
file_path_export=Path("../../data/processed/Housing_cleaned2.csv")
FL_CA_df.to_csv(file_path_export,index=False)


# In[13]:


# Load the table to database
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic"
engine = create_engine(db_string)
FL_CA_df.to_sql(name='housing_cleaned_data', con=engine, if_exists='replace')


# In[ ]:




