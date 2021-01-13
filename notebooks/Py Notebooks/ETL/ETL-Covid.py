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


# Read the raw covid data file
file_path = Path("../../data/raw/covid/RAW_us_confirmed_cases.csv")
covid_df = pd.read_csv(file_path)
print(covid_df.shape)
covid_df.head()


# ## ETL - Transform Process
# Keep only FL and CA data and then drop unnecessary columns

# In[3]:


# Check the column names
print(covid_df.columns.values)


# In[4]:


# Drop the unnecessary columns
covid_df.drop(columns=["UID","iso2","iso3","FIPS","code3","Country_Region","Combined_Key"],inplace=True)
print(covid_df.shape)


# In[5]:


# Rename Province_State to State and Admin2 to County
covid_df.rename(columns={"Province_State":"State","Admin2":"County"},inplace=True)
print(covid_df.shape)
covid_df.head()                        


# In[6]:


# Filter rows with state FL and CA only
state = ["Florida", "California"]
state_covid_data = covid_df[covid_df.State.isin(state)]
state_covid_data.dropna(axis=0,how="any",inplace=True)
print(state_covid_data.shape)
state_covid_data.head()


# In[7]:


## Dataframe for State and County. 
df_counties = pd.DataFrame()
df_counties["State"] = state_covid_data["State"]
df_counties["County"] = state_covid_data["County"]
print(df_counties.shape)
df_counties.head()


# Looks like the covid numbers are cumulative numbers added each day. So in order to get the monthly numbers
# we will just take the numbers from the last day of month

# In[8]:


## Get last day of month numbers
eom_dates =  []
import calendar
year = 2020
for i in range(1,12):
    #print(f'{i}/{calendar.monthrange(year,i)[1]}/{year}')
    eom_dates.append(str(i)+"/"+str(calendar.monthrange(year,i)[1])+"/"+str(year)[-2:])

print(eom_dates)


# In[9]:


## Extract the columns specific to end of month
df_dates = pd.DataFrame()
for key in eom_dates:
    df_dates[key] = state_covid_data[key]

df_dates


# To get covid numbers for each month (and not cumulative), we will subtract M1 from M2

# In[10]:


df_dates_unique=df_dates.diff(axis=1)
df_dates_unique.iloc[:, 0]=df_dates.iloc[:, 0]
print(df_dates_unique.shape)
df_dates_unique.head()


# In[11]:


# Merge the date columns with state and county columns from the trimmed dataframe
df_covid_final = pd.concat([df_counties, df_dates_unique], axis=1)
print(df_covid_final.shape)
df_covid_final.head()


# Check columns types and if any null values

# In[12]:


df_covid_final.dtypes


# In[13]:


# Change the first column to float64 to be consistent with other columns
df_covid_final['1/31/20'] = df_covid_final['1/31/20'].astype(float) 
df_covid_final.dtypes


# In[14]:


# Check if any null values present
df_covid_final.isnull().sum()


# Everything looks good, Tranpose the data now.

# In[15]:


## Convert the Date columns to rows
FL_CA_covid_df = pd.melt(df_covid_final, id_vars=["State","County"], 
                  var_name="Date", value_name="Covid_Cummulative_Numbers")


# In[16]:


print(FL_CA_covid_df.shape)


# In[17]:


FL_CA_covid_df.sample(n=10)


# In[18]:


FL_CA_covid_df.sample(n=10)


# In[19]:


# Drop the Rows where any field is null or blank
FL_CA_covid_df.dropna(axis=0)


# In[20]:


# Convert the date field to Datetime field
FL_CA_covid_df["Date"] = pd.to_datetime(FL_CA_covid_df["Date"])
FL_CA_covid_df.sample(n=10)


# In[21]:


# Ensure Avg_Price field is of type integer 
print(FL_CA_covid_df["Covid_Cummulative_Numbers"].dtype)
FL_CA_covid_df['Covid_Cummulative_Numbers'] = FL_CA_covid_df['Covid_Cummulative_Numbers'].astype('Int64')
FL_CA_covid_df.sample(n=10)


# ## ETL - Load data 
# write the processed data to CSV and load to database

# In[22]:


#Export the data to CSV
file_path_export=Path("../../data/processed/covid_cleaned.csv")
FL_CA_covid_df.to_csv(file_path_export,index=False)


# In[23]:


# Load the table to database
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic"
engine = create_engine(db_string)
FL_CA_covid_df.to_sql(name='covid_cleaned', con=engine, if_exists='replace')

