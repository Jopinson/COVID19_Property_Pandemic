# COVID19_Property_Pandemic

## SEGMENT 2 Deliverables

### Presentation
The presentation currently is being worked on in [GoogleSlides](https://docs.google.com/presentation/d/1LOd4DVS-7EgPJrmVBYNyDwg0x_EM48Xhtd77qu3fqFc/edit?usp=sharing)

### Machine Learning
The dataset that we had for housing data contained only dates and average housing prices from 1996-2020 for regions across USA. Since, the goal was to predict the 2020 housing prices and compare the predicted price with actual price we went with UniVariate Linear Regression model. <br>
The feature of our model is the Monthly dates and the target is average house prices. <br>
The machine learning model notebook can be found [here](notebooks/modeling/Covid-Housing-ML.ipynb). <br>
This file right now contains model and prediction for Los Angeles area. We are planning to create similar models for San Jose, CA; Lakeland, FL and Miami, FL.<br>
More details on machine learning is documented in the presentation slides.

### Dashboard
We are using Tableau for creating dashboard. The link to the Tableau:<br>
[https://public.tableau.com/profile/angela.silveira#!/vizhome/COVID19CAFLByCounty/Dashboard1](https://public.tableau.com/profile/angela.silveira#!/vizhome/COVID19CAFLByCounty/Dashboard1)<br>

### Database
We decided to use POSTGRES as our database. The database requires the following One-Time setup:<br>
1. Install POSTGRES (pgAdmin) version 4.24 or higher
2. Create a Database with name 'covid_property_pandemic". *(In Segment 3, planning to create a script that will create the database. )*<br>
3. Create user 'postgres', if not already present.
4. Create a config.py file and add db_password = <yourpassword> to the file. 
5. The tables in the database are populated through ETL process. The 3 ETL notebooks are available at: <br> [housing_cleaned_data table from Housing ETL](notebooks/ETL/ETL-Housing.ipynb)<br>[covid_cleaned table from COVID ETL](notebooks/ETL/ETL-Covid.ipynb)<br>[city_county_mapping table from City County Mapping ETL](notebooks/ETL/ETL_City_County_Map.ipynb)

The Housing data has City Name and State Name as columns and the COVID data has County Name and State Name as columns. So we used Join statements from city_county_mapping table and housing_cleaned_data table to add county column for housing data. The sql for join statements is available [here](data/queries/proj_tables.sql).  <br><br>
The ERD information can be found [here](data/erd/ERD_Diagram.png)



## Overview
We are attempting to create a machine learning model that can predict housing prices. Using [this dataset from Zillow](https://www.zillow.com/research/data/) we are going to create a pricing model with data from 2010-2019 and use it to predict 2020 prices. Using our pricing model, we will atempt to see if Covid-19 has affected the housing market compared to the predicted values. We will be using a wide range of programming and data visualization tools including but not limited to Javascript, postgres SQL, Python, and Tableau.


### Communication Protocols 
- Private Slack Channel
- [Google Slides](https://docs.google.com/presentation/d/1LOd4DVS-7EgPJrmVBYNyDwg0x_EM48Xhtd77qu3fqFc/edit?usp=sharing)
- Group Meetings on Tuesdays, Thursdays, and Sundays for progress checks
- Github Projects
- Github Issues


### Folder Structure
- [data](data) - Contains Raw data, Processed Data, ERDs and SQL Queries
- [models](models) - Contains trained models
- [notebooks](notebooks) - Contains notebooks for ETL, data preprocessing, modeling
- [reports](reports) - Contains final presentation slides, HTML / PDF reports, images required for reports and README 


We came together to code things a single programmer can't possibly code by themselves, we are Earth's Mightiest Coders!
![1542272348-the-avengers](https://user-images.githubusercontent.com/68392225/101806789-fead0c00-3ad9-11eb-91bc-6704c91e43f8.jpg)
