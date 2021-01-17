# 1. Extract the csv file directly from the websites
# 2. Transform the data into required format for machine learning model
# 3. Load data into database

import warnings; warnings.simplefilter('ignore')
import pandas as pd
from pathlib import Path
import calendar
from config import db_password
from sqlalchemy import create_engine

proj_root_dir = Path(__file__).resolve().parents[0]
print(proj_root_dir)


def extract_covid_data():
    # COVID data file was downloaded from https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university?select=RAW_us_confirmed_cases.csv
    covid_data_file = f"{proj_root_dir}/data/raw/covid/RAW_us_confirmed_cases.csv"
    covid_df = pd.read_csv(covid_data_file)
    print(covid_df.shape)
    return covid_df

def extract_housing_data():
    # Housing data was downloaded from https://www.zillow.com/research/data/ as 11/30/2020
    # Under Home Values > Data Type = ZHVI All Homes (SFR, Condo/Co-op) Time Series, Smoothed, Seasonally Adjusted($)
    # Geography = Metro & U.S
    zillow_data_file = f"{proj_root_dir}/data/raw/zillow_data/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_mon.csv"
    zillow_df = pd.read_csv(zillow_data_file)
    
    return zillow_df

def extract_county_city_mapping():
    # The uscities data file is downloaded from free database from https://simplemaps.com/data/us-cities

    us_cities_file = f"{proj_root_dir}/data/raw/uscities.csv"
    city_df = pd.read_csv(us_cities_file)
    return city_df

def transform_covid_data(covid_df):
    # Drop the unnecessary columns
    covid_df.drop(columns=["UID","iso2","iso3","FIPS","code3","Country_Region","Combined_Key"],inplace=True)
    
    # Rename Province_State to State and Admin2 to County
    covid_df.rename(columns={"Province_State":"State","Admin2":"County"},inplace=True)
    

    # Filter rows with state FL and CA only
    state = ["Florida", "California"]
    state_covid_data = covid_df[covid_df.State.isin(state)]
    state_covid_data.dropna(axis=0,how="any",inplace=True)

    ## Dataframe for State and County. 
    df_counties = pd.DataFrame()
    df_counties["State"] = state_covid_data["State"]
    df_counties["County"] = state_covid_data["County"]
    

    # the covid numbers are cumulative numbers added each day. 
    # So in order to get the monthly numbers we will just take the numbers from the last day of month
    eom_dates =  []

    year = 2020
    for i in range(1,12):
        eom_dates.append(str(i)+"/"+str(calendar.monthrange(year,i)[1])+"/"+str(year)[-2:])

    ## Extract the columns specific to end of month
    df_dates = pd.DataFrame()
    for key in eom_dates:
        df_dates[key] = state_covid_data[key]
    
    # To get covid numbers for each month (and not cumulative), we will subtract M1 from M2
    df_dates_unique=df_dates.diff(axis=1)
    df_dates_unique.iloc[:, 0]=df_dates.iloc[:, 0]
    

    # Merge the date columns with state and county columns from the trimmed dataframe
    df_covid_final = pd.concat([df_counties, df_dates_unique], axis=1)
    

    # Change the first column to float64 to be consistent with other columns
    df_covid_final['1/31/20'] = df_covid_final['1/31/20'].astype(float) 

    ## Convert the Date columns to rows
    FL_CA_covid_df = pd.melt(df_covid_final, id_vars=["State","County"], var_name="Date", value_name="Covid_Cummulative_Numbers")

    # Drop the Rows where any field is null or blank
    FL_CA_covid_df.dropna(axis=0)

    # Convert the date field to Datetime field
    FL_CA_covid_df["Date"] = pd.to_datetime(FL_CA_covid_df["Date"])

    # Ensure Avg_Price field is of type integer 
    FL_CA_covid_df['Covid_Cummulative_Numbers'] = FL_CA_covid_df['Covid_Cummulative_Numbers'].astype('Int64')
    
    return FL_CA_covid_df    

def transform_housing_data(zillow_df):
    # Filter rows with state FL and CA only
    state = ["FL", "CA"]
    state_df = zillow_df[zillow_df.StateName.isin(state)]
    state_df.dropna(axis=0,how="any",inplace=True)

    # Drop the unnecessary columns
    state_dropped_df =state_df.drop(columns=["RegionID", "SizeRank", "RegionType"])

    ## Convert the Date columns to rows
    FL_CA_df = pd.melt(state_dropped_df, id_vars=["RegionName","StateName"], var_name="Date", value_name="Avg_Price")
    FL_CA_df.set_index("RegionName")

    # Drop the Rows where any field is null or blank
    FL_CA_df.dropna(axis=0)

    # Convert the date field to Datetime field
    FL_CA_df["Date"] = pd.to_datetime(FL_CA_df["Date"])
    
    # Remove State from the RegionName
    FL_CA_df['RegionName'] = FL_CA_df.apply(lambda x:x['RegionName'][:-4], axis=1)

    # Ensure Avg_Price field is of type integer 
    FL_CA_df['Avg_Price'] = FL_CA_df['Avg_Price'].astype('Int64')

    FL_CA_df.rename(columns={"RegionName":"region_name","StateName":"state_name","Date":"date","Avg_Price":"avg_price"},inplace=True)

    return FL_CA_df

def transform_city_county_mapping(city_df):
    # Drop the columns not required
    city_df.drop(columns=["city_ascii","county_fips","lat","lng","population","density","source","military","incorporated","timezone","ranking","zips","id"],inplace=True)

    # Get cities and counties only for FL and CA
    city_df_ca_fl = city_df.loc[city_df["state_id"].isin(['FL','CA'])]
    
    return city_df_ca_fl

def load_covid_data(FL_CA_covid_df):
    #Export the data to CSV
    file_path_export=Path(f"{proj_root_dir}/data/processed/covid_cleaned_one.csv")
    FL_CA_covid_df.to_csv(file_path_export,index=False)


    # Load the table to database
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic_one"
    engine = create_engine(db_string)
    FL_CA_covid_df.to_sql(name='covid_cleaned', con=engine, if_exists='replace')        
    return

def load_housing_data(cleaned_housing_df):
    #Export the data to CSV
    file_path_export=Path(f"{proj_root_dir}/data/processed/housing_cleaned_one.csv")
    cleaned_housing_df.to_csv(file_path_export,index=False)


    # Load the table to database
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic_one"
    engine = create_engine(db_string)
    cleaned_housing_df.to_sql(name='housing_cleaned_data', con=engine, if_exists='replace')        
    return
    

def load_city_county_map_data(city_df_ca_fl):
    #Export the data to CSV
    file_path_export=Path(f"{proj_root_dir}/data/processed/city_county_mapping.csv")
    city_df_ca_fl.to_csv(file_path_export,index=False)

    # Load the table to database
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic_one"
    engine = create_engine(db_string)
    city_df_ca_fl.to_sql(name='city_county_mapping', con=engine, if_exists='replace')        
    
    return

#ETL Covid data
covid_df = extract_covid_data()
cleaned_covid_df = transform_covid_data(covid_df)
load_covid_data(cleaned_covid_df)
print("ETL process of COVID data complete")

#ETL Housing data
zillow_df = extract_housing_data()
cleaned_housing_df = transform_housing_data(zillow_df)
load_housing_data(cleaned_housing_df) 
print("ETL process of Housing data complete")

# ETL City-County Mapping
city_df = extract_county_city_mapping()
city_county_df = transform_city_county_mapping(city_df)
load_city_county_map_data(city_county_df)
print("ETL process of City County Mapping complete") 