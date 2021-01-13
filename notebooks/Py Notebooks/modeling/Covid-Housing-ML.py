#!/usr/bin/env python
# coding: utf-8

# In[32]:


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
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import numpy as np


# In[2]:


import os, sys
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


# In[83]:


train_df = base_df.loc[(base_df["Date"] <'2020-01-01')]
train_df


# In[80]:


# Separate actual 2020 sales so 2010-2019 can be used in ML predictions
actual_df = base_df.loc[(base_df["Date"] > '2020-01-01')]
actual_df


# In[108]:


# Create datesframes for each state + top COVID cities for each (CA = LA/FL = Miami)
FL_base_df = base_df.loc[(base_df["StateName"] == "FL")]
FL_actual_df = actual_df.loc[(actual_df["StateName"] == "FL")]
Miami_base_df = base_df.loc[(base_df["RegionName"] == "Miami-Fort Lauderdale")]
Miami_actual_df = actual_df.loc[(actual_df["RegionName"] == "Miami-Fort Lauderdale")]
CA_base_df = base_df.loc[(base_df["StateName"] == "CA")]
CA_actual_df = actual_df.loc[(actual_df["StateName"] == "CA")]
LA_base_df = train_df.loc[(train_df["RegionName"] == "Los Angeles-Long Beach-Anaheim")]
LA_actual_df = actual_df.loc[(actual_df["RegionName"] == "Los Angeles-Long Beach-Anaheim")]
LA_base_df


# In[104]:


nineteen_df = LA_base_df.loc[(base_df["Date"] < '2011-01-01')]
nineteen_df


# In[ ]:





# In[28]:


regionsFL = FL_base_df.RegionName.nunique()
regionsCA = CA_base_df.RegionName.nunique()
print(f"FL Regions: {regionsFL}, CA Regions: {regionsCA}")


# In[85]:


# Graph home sales from 2010-2019 for LA
fig = px.scatter(LA_base_df, x="Date", y="Avg_Price", color="RegionName", hover_data=['RegionName'])
fig.show()
#plt.savefig("..\reports\images\LA2010-2019Sales.png")
fig1 = px.line(LA_base_df, x="Date", y="Avg_Price", color="RegionName", width=800)
fig1.show()


# In[86]:


# Graph home sales for 2010-2019 for Miami
fig = px.scatter(Miami_base_df, x="Date", y="Avg_Price", color="RegionName", hover_data=['RegionName'])
fig.show()
#plt.savefig("..\reports\images\Miami2010-2019Sales.png")
fig1 = px.line(Miami_base_df, x="Date", y="Avg_Price", color="RegionName", width=800)
fig1.show()


# # California Data Machine Learning - Linear Regression Model

# In[109]:


# Declaring features and targets - LA
LA_base_df['Date'] = pd.to_datetime(LA_base_df['Date'])
LA_base_df['Date']=LA_base_df['Date'].map(dt.datetime.toordinal)
X = LA_base_df.Date.values.reshape(-1,1)
y= LA_base_df.Avg_Price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
LA_base_df


# In[113]:


#instantized and fit data to model and predictions
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(LA_actual_df['Date'].map(dt.datetime.toordinal).values.reshape(-1,1))
ml_df = pd.DataFrame({'Actual': y_test, 'Predicted':  y_pred})
ml_df.reset_index(inplace=True)


# In[114]:


model = LinearRegression()
model.fit(X_train, y_train)
LA_actual_df['Date'] = pd.to_datetime(LA_actual_df['Date'])
LA_actual_df['Date']=LA_actual_df['Date'].map(dt.datetime.toordinal)
X = LA_actual_df.Date.values.reshape(-1,1)
y_pred = model.predict(X)


# In[100]:


X


# In[89]:


#Print regression line
plt.scatter(X, y)
plt.plot(X_test, y_pred, color="red")
plt.show()


# In[90]:


y_pred


# In[91]:


#Print the coeffiecient and intercept of the nodel
print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test,y_pred))


# In[116]:


# Compare y_pred array with LA_actual_df
fig = go.Figure()
x_dates = LA_actual_df['Date']
y_actual = LA_actual_df['Avg_Price']
fig.add_trace(go.Scatter(x=x_dates, y=y_actual, mode='markers', name='Actual 2020 Average Sale Price'))
fig.add_trace(go.Scatter(x=x_dates, y=y_pred, mode='lines', name='Model Predicted'))
fig.show()


# # Florida Data Machine Learning - Linear Regression Model

# In[50]:


# Declaring features and targets - Miami
Miami_base_df['Date'] = pd.to_datetime(Miami_base_df['Date'])
Miami_base_df['Date']=Miami_base_df['Date'].map(dt.datetime.toordinal)
X = Miami_base_df.Date.values.reshape(-1,1)
y= Miami_base_df.Avg_Price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[51]:


#instantized and fit data to model and predictions
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mlfl_df = pd.DataFrame({'Actual': y_test, 'Predicted':  y_pred})
mlfl_df


# In[57]:


#Print regression line
plt.scatter(X, y)
plt.plot(X, y_pred, color="red")
plt.show()


# In[53]:


#Print the coeffiecient and intercept of the nodel
print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test,y_pred))


# In[56]:


# Compare y_pred array with Miami_actual_df
fig = go.Figure()
x_dates = Miami_actual_df['Date']
y_actual = Miami_actual_df['Avg_Price']
fig.add_trace(go.Scatter(x=x_dates, y=y_actual, mode='markers', name='Actual 2020 Average Sale Price'))
fig.add_trace(go.Scatter(x=x_dates, y=y_pred[:10], mode='lines', name='Model Predicted'))
fig.show()


# In[ ]:




