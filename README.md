# End to End Machine Learning model to predict Housing Prices during COVID-19 


[Link to Github Pages](https://jopinson.github.io/COVID19_Property_Pandemic/)<br>
[Link to Google Slides](https://docs.google.com/presentation/d/1LOd4DVS-7EgPJrmVBYNyDwg0x_EM48Xhtd77qu3fqFc/edit#slide=id.p1)<br>
## Table of Contents
[Overview](#overview) <br>
[Tools and Technologies](#tools-and-technologies)<br>
[Data](#data) <br>
[Data Observations](#data-observations)<br>
[Machine Learning Model](#machine-learning-model) <br>
[Installation](#installation)<br>
[Folder Structure](#folder-structure)<br>
[Team](#team)<br><br>

## Overview
After the housing market crash of 2007-2008, the house prices in the last decade rose steadily. And then came 2020. As the COVID-19 cases started spiking in late March, the local and state governments started lockdowns to control the spread of virus. <br>
With the businesses closed, there was a wide spread fear of another recession. But surprisingly the housing market wasn't affected much. <br>
So as part of this project we wanted to analyze if COVID-19 had any impact on house prices. To limit the scope of the project, we decided to look at **California** and **Florida** markets as these states reported highest number of COVID cases.
<br><br>
***One to two line summary of the analysis results***<br><br>
[back to ToC](#table-of-contents)
## Tools and Technologies
**Pandas**<br>
**SkLearn**<br>
**Numpy**<br>
**Tableau**<br>
**PostGres DB**<br><br>
[back to ToC](#table-of-contents)
## Data
### Data Source
The data was collected from following sources as of 11/30/2020.<br><br>
COVID Data - [https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university?select=RAW_us_confirmed_cases.csv](https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university?select=RAW_us_confirmed_cases.csv). This data file is downloaded [here](data/raw/covid/RAW_us_confirmed_cases.csv).<br><br>
Housing Data - [ZHVI All Homes (SFR, Condo/Co-op) Time Series, Smoothed, Seasonally Adjusted($) data for Metro & U.S](https://www.zillow.com/research/data/). This data file is downloaded [here](data/raw/zillow_data/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_mon.csv)<br><br>

Since COVID data was aggregated by Counties and the Housing data was aggregated by Metro regions, we also needed city and county mapping data. This we obtained from free database from [https://simplemaps.com/data/us-cities](https://simplemaps.com/data/us-cities). The data file is downloaded [here](data/raw/uscities.csv).

### ETL Process
ETL code for all 3 raw data files can be reviewed [here](notebooks/ETL/). For all 3 raw data files, the code cleans and saves the processed data file in [data/processed](data/processed/) folder as well as loads the processed data to PostGres tables.<br> The details of the PostGres tables can be found [here](data/erd/ERD_Diagram.png).<br><br> 
[COVID-ETL](notebooks/ETL/ETL-Covid.ipynb):<br>
The raw covid data file has covid cummulative numbers for each day from 1/1/2020 to 11/30/2020. As part of transformation process, following steps were performed - <br> 
1. Filter the data for counties in California and Florida states only and removed unnecessary columns
2. The covid numbers were aggregated by month for each county 
3. The wide format was converted to long format. <br><br> 

[Housing-ETL](notebooks/ETL/ETL-Housing.ipynb):<br><br>
The raw housing data file contained aggregated housing prices by month of each Metro region in US from 01/01/1996 to 11/30/2020. As part of transformation process, following steps were performed - <br>
1. Filter the data for metro regions in California and Florida states only and removed unnecessary columns.
2. The wide format was converted to long format
3. The metro region name contained two letter state code as well. The state code was removed from the region name.<br><br>

[city-county-mapping](notebooks/ETL/ETL_City_County_Map.ipynb):<br>As part of ETL process for city county mapping, the unnecessary columns were removed from [uscities.csv](data/raw/uscities.csv).

[back to ToC](#table-of-contents)<br>
## Data Observations

Based on the below visualizations, following observations can be made:<br>
1. Los Angeles has the highest number of cases in CA. Santa Clara had the second highest cases in March but the spread was controlled slowly over following months.
2. Miami-Dade county has the highest numbers of cases in FL. Polk county started with low number of cases, but in the following months it saw increase in the COVID case numbers.
3. The housing market was at Peak during 2008. But once the market crashed it took some time for prices to stabilize. **Hence we are going to look at house prices only from 2010 - 2020 for our analysis and machine learning model training.**
4. House prices for Los Angeles, Miami Fort Lauderdale and Lakeland regions rose steadily from 2010 to 2020. 
5. San Jose house prices also rose steadily from 2010 to 2020, but the region saw a little dip in house prices in 2019.
### COVID Visualizations
Top 10 Counties in California with highest COVID cases <br>
![](reports/images/COVID19CasesByCounty%20-%20CA.png)<br><br>
Top 10 Counties in Florida with highest COVID cases <br>
![](reports/images/COVID19CasesByCounty%20-%20FL.png)<br><br>
### Housing Visualizations
![](reports/images/ca_housing_prices_1996_2020.png)<br>
<br>
![](reports/images/fl_housing_prices_1996_2020.png)<br>
<br>

**Average house prices from 2010 - 2020 in Los Angeles and Santa Clara (San Jose region) counties in California**<br>
![](images/Los%20Angeles-Long%20Beach-Anaheim_avg_price.png)  ![](images/San%20Jose_avg_price.png)<br>

**Average house prices from 2010 - 2020 in Miami-Dade and Polk (Lakeland region) counties in Florida**<br>
![](images/Miami-Fort%20Lauderdale_avg_price.png)  ![](images/Lakeland_avg_price.png)<br>




[back to ToC](#table-of-contents)<br>
## Machine Learning Model
The dataset that we had for housing data contained only dates and average housing prices from 1996-2020 for regions across USA. Since, the goal was to predict the 2020 housing prices and compare the predicted price with actual price we went with UniVariate Linear Regression model. <br>
The feature of our model is the Monthly dates and the target is average house prices. <br>
The machine learning model notebook can be found [here](notebooks/modeling/Covid-Housing-ML.ipynb). <br>
This file right now contains model and prediction for Los Angeles area. We are planning to create similar models for San Jose, CA; Lakeland, FL and Miami, FL.<br>
More details on machine learning is documented in the presentation slides.

### Choice of model and metrics

### Data Preprocessing

### Predictive analysis and visualization




[back to ToC](#table-of-contents)<br>
## Installation
### Prerequisites

1. Python 3.8.5
2. PostGres DB (pgAdmin 4.24)
<br><br>

### Activate conda environment
conda create -name CovidHousing python=3.8<br>
conda activate CovidHousing
<br><br>

### One Time Setup
1. Clone the git repo = <project_root_folder>
2. create `config.py` under project_root_folder; add db_password = <your_postgres_db_password> to file and save.  
3. Start the pgAdmin server
4. pip install -r requirements.txt
5. Run python init_db.py
<br><br>

### Run

Run python files in the following order:
1. python etl_data.py
2. python join_tables.py
3. python ml_model_train.py<br>
    a. The graphs from the model are saved in [images] folder<br>
    b. The results from the model are saved in csv files under [data/processed/](../data/prcessed/)<br>
    > i. [result metrics](../data/processed/model_metrics.csv)<br>
    ii. [2020 Actual Vs Predicted Prices](../data/processed/ml_2020_act_pred.csv)<br>
<br>

[back to ToC](#table-of-contents)<br>

### Folder Structure
- [data](data/) - Contains Raw data, Processed Data, ERDs and SQL Queries
- [models](models/) - Contains trained models
- [notebooks](notebooks/) - Contains notebooks for ETL, data preprocessing, modeling
- [reports](reports/) - Contains final presentation slides, HTML / PDF reports, images required for reports and README
- [images](images/) - Contains images required for index.html
- [html](html/) - Contains html code for embedding Tableau dashboards in index.html

[back to ToC](#table-of-contents)
## Team
We came together to code things a single programmer can't possibly code by themselves, we are Earth's Mightiest Coders!<br>
[Angela Silveira](https://github.com/Angela-Silveira)<br>
[Duane Conley](https://github.com/Duane93)<br>
[Harishawn Ramrup](https://github.com/ramruph)<br>
[Johnathon Pinson](https://github.com/Jopinson)<br>
[Nitu Singh](https://github.com/vnitusingh)<br>

![1542272348-the-avengers](https://user-images.githubusercontent.com/68392225/101806789-fead0c00-3ad9-11eb-91bc-6704c91e43f8.jpg)
