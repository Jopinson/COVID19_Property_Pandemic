#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd
from path import Path
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sqlalchemy import create_engine
import psycopg2 
# import the psycopg2 database adapter for PostgreSQL
from psycopg2 import connect, extensions, sql
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import numpy as np


# In[2]:


import os,sys
parentdir = Path(os.path.abspath("../.."))
sys.path.insert(0,parentdir)

# Get the DB password
from config import db_password


# In[3]:


# Create Engine for covid_property_pandemic DB
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic"
engine = create_engine(db_string)
conn = engine.connect()


# In[4]:


# Get Data from Housing Table

sql_str = 'SELECT * FROM "FL_CA_Housing_Data"'

df = pd.read_sql(sql_str,conn)
print(df.shape)
df.sample(n=10)


# In[5]:


# Remove pre-2010 recession sales data
base_df = df[df["Date"] > '2010-01-01']
base_df


# In[6]:


train_df = base_df.loc[(base_df["Date"] <'2020-01-01')]
train_df


# In[7]:


# Separate actual 2020 sales so 2010-2019 can be used in ML predictions
actual_df = base_df.loc[(base_df["Date"] > '2020-01-01')]
actual_df


# In[8]:


# Create datesframes for each state + top COVID cities for each (CA = LA/FL = Miami)
FL_base_df = base_df.loc[(base_df["StateName"] == "FL")]
FL_actual_df = actual_df.loc[(actual_df["StateName"] == "FL")]

Miami_base_df = train_df.loc[(train_df["RegionName"] == "Lakeland")]
Miami_actual_df = actual_df.loc[(actual_df["RegionName"] == "Lakeland")]



CA_base_df = base_df.loc[(base_df["StateName"] == "CA")]
CA_actual_df = actual_df.loc[(actual_df["StateName"] == "CA")]

LA_base_df = train_df.loc[(train_df["RegionName"] == "Los Angeles-Long Beach-Anaheim")]
LA_actual_df = actual_df.loc[(actual_df["RegionName"] == "Los Angeles-Long Beach-Anaheim")]
#LA_base_df


MIA_base_df = train_df.loc[(train_df["RegionName"] == "Lakeland")]
MIA_actual_df = actual_df.loc[(actual_df["RegionName"] == "Lakeland")]
MIA_base_df


# In[9]:


MIA_actual_df


# In[10]:


regionsFL = FL_base_df.RegionName.nunique()
regionsCA = CA_base_df.RegionName.nunique()
print(f"FL Regions: {regionsFL}, CA Regions: {regionsCA}")


# In[11]:


# Graph home sales from 2010-2019 for LA
fig = px.scatter(LA_base_df, x="Date", y="Avg_Price", color="RegionName", hover_data=['RegionName'])
fig.show()
#plt.savefig("..\reports\images\LA2010-2019Sales.png")
fig1 = px.line(LA_base_df, x="Date", y="Avg_Price", color="RegionName", width=800)
fig1.show()


# In[12]:


# Graph home sales for 2010-2019 for Miami
fig = px.scatter(Miami_base_df, x="Date", y="Avg_Price", color="RegionName", hover_data=['RegionName'])
fig.show()
#plt.savefig("..\reports\images\Miami2010-2019Sales.png")
fig1 = px.line(Miami_base_df, x="Date", y="Avg_Price", color="RegionName", width=800)
fig1.show()


# In[13]:


# Declaring features and targets - LA
MIA_base_df['Date'] = pd.to_datetime(MIA_base_df['Date'])
MIA_base_df['Date']= MIA_base_df['Date'].map(dt.datetime.toordinal)
X = MIA_base_df.Date.values.reshape(-1,1)
y= MIA_base_df.Avg_Price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[14]:


#instantized and fit data to model and predictions
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_base = model.predict(X_test)
ml_df = pd.DataFrame({'Actual': y_test, 'Predicted':  y_pred_base})
ml_df.reset_index(inplace=True)


ml_df


# In[15]:


model = LinearRegression()
model.fit(X_train, y_train)
MIA_actual_df['Date'] = pd.to_datetime(MIA_actual_df['Date'])
MIA_actual_df['Date']=MIA_actual_df['Date'].map(dt.datetime.toordinal)
X = MIA_actual_df.Date.values.reshape(-1,1)
y_pred = model.predict(X)


# In[16]:


y_pred


# In[17]:


#Print regression line
plt.scatter(X, y_pred)
plt.plot(X, y_pred, color="red")
plt.show()


# In[18]:


#Print the coeffiecient and intercept of the nodel

print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_base))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_base))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_base)))
print('R2 Score:', r2_score(y_test,y_pred_base))


# In[19]:



# Compare y_pred array with MIA_actual_df
fig = go.Figure()
x_dates = MIA_actual_df['Date']
y_actual = MIA_actual_df['Avg_Price']
fig.add_trace(go.Scatter(x=x_dates, y=y_actual, mode='markers', name='Actual 2020 Average Sale Price'))
fig.add_trace(go.Scatter(x=x_dates, y=y_pred, mode='lines', name='Model Predicted'))
fig.show()


# In[ ]:




