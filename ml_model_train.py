import warnings; warnings.simplefilter('ignore')

import pandas as pd
import datetime as dt
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from config import db_password

proj_root_dir = Path(__file__).resolve().parents[0]

# Create Engine for covid_property_pandemic DB
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/covid_property_pandemic_one"
engine = create_engine(db_string)
conn = engine.connect()

# Get Data from Housing Table
sql_str = 'SELECT * FROM fl_ca_housing_data'
df = pd.read_sql(sql_str,conn)

def convert_ordinal_to_date(data):
    x = data.reshape(1,-1)
    ret_dates = []
    array_size = x.shape[1]
    for i in range(array_size):
        ret_dates.append(dt.datetime.fromordinal(x[0][i]).strftime("%Y-%m-%d"))
    
    return ret_dates

def lnr_model(df,city):
    results = {'city':city}
    
    # Remove pre-2010 recession sales data and create a base df for given city 
    base_df = df[(df["date"] > '2010-01-01') & (df["region_name"] == city)]
    base_df['ordinal_date'] = base_df['date'].map(dt.datetime.toordinal) 
    
    # Plot the base dataframe
    fig = plt.figure()
    fig.subplots_adjust(top=0.8,left=0.3)
    ax = fig.add_subplot()
    line, = ax.plot(base_df['date'].values,base_df['avg_price'].values)
    ax.set_xlabel('Year')
    ax.set_ylabel('Avg House Prices (USD)')
    ax.set_title(f"Average House Prices in {city} 2010-2020")
    plt.show
    plt.savefig(f"{proj_root_dir}/images/{city}_avg_price.png")
    
    # Create training set from 2010-2019;
    train_df = base_df.loc[(base_df["date"] <'2020-01-01')]
    
    # Create the 2020 dataset to verify
    df_2020 = base_df.loc[(base_df["date"] > '2020-01-01')]
    
    # Create Feature and Target for training and testing the model from 2010-2019 DataSet
    X = train_df.ordinal_date.values.reshape(-1,1)
    y =  train_df.avg_price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    #instantized and fit data to model and predictions
    model = LinearRegression().fit(X_train, y_train)
    
    results['Model R2 Score'] = model.score(X_train, y_train)
    results['Model Coefficient'] = model.coef_
    results['Model Intercep'] = model.intercept_
    
    # Print the model metrics 
    #print(f"Model R2 Score, Coefficient and Intercept values for '{city}'")
    #print("*******************************************************************")
    #print(f"Coefficient: {results['Model Coefficient']}, Intercept: {results['Model Intercep']}")
    #print(f'Model R2 Score: ',results['Model R2 Score'])
    #print("\n\n\n")
    
    # Test Prediction with 2019 data
    y_test_pred = model.predict(X_test)
    
    #Print regression line
    fig2 = plt.figure(figsize=(8, 4))
    ax = plt.axes()
    ax.scatter(convert_ordinal_to_date(X), y)
    ax.plot(convert_ordinal_to_date(X_test), y_test_pred, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg Price (USD)')
    ax.axis('tight')
    ax.set_title(f"Regression line for city '{city}'")
    plt.show()
    fig2.savefig(f"{proj_root_dir}/images/{city}_reg_line.png")
    
    # Print the actual and predicted prices in a dataframe
    #print(f"The Actual Vs Predicted House Prices on Test Data for '{city}'")
    #print("*******************************************************************")
    ml_df_test_set = pd.DataFrame({'date':convert_ordinal_to_date(X_test),'test_set_actual_prices': y_test, 'test_set_predicted_prices':  y_test_pred})
    ml_df_test_set.set_index('date',inplace=True)
    ml_df_test_set['city'] = city
    #print(ml_df_test_set)
    #print("\n\n\n")
    
    results['TestSet MAE Value'] = metrics.mean_absolute_error(y_test, y_test_pred)
    results['TestSet MSE Value'] = metrics.mean_squared_error(y_test, y_test_pred)
    results['TestSet RMSE Value'] = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    #Print the MAE, MSE and RMSE values
    #print(f"MAE, MSE and RMSE on Test Data for '{city}'")
    #print("*******************************************************************")
    #print('Mean Absolute Error:', results['TestSet MAE Value'])
    #print('Mean Squared Error:', results['TestSet MSE Value'] )
    #print('Root Mean Squared Error:', results['TestSet RMSE Value'])
    #print("\n\n\n")
    
    # TEST SET GRAPH
    ax3 = ml_df_test_set.plot(figsize = (15,8))
    plt.legend(bbox_to_anchor=(0.5,0.5), loc='upper left', ncol=1)
    plt.xlabel("Month", fontsize = 16)
    plt.ylabel("Average Sale Price (USD)")
    plt.title(f"Average Sale Price on Test Data: Actual Vs Predicted for city '{city}'")   
    plt.show()
    ax3.get_figure().savefig(f"{proj_root_dir}/images/{city}_test_set_act_pred.png")
    
    
    # Verify Prediction with 2020 data
    X_2020 = df_2020.ordinal_date.values.reshape(-1,1)
    y_2020_actual_prices =  df_2020.avg_price
    y_2020_pred_prices = model.predict(X_2020)
    ml_2020_df = pd.DataFrame({'date':convert_ordinal_to_date(X_2020),'actual_2020_house_prices': y_2020_actual_prices, 'predicted_2020_house_prices':  y_2020_pred_prices})
    ml_2020_df.set_index('date',inplace=True)
    ml_2020_df['city'] = city
    #print(f"The Actual Vs Predicted House Prices on 2020 data for '{city}'")
    #print("*******************************************************************")
    #print(ml_2020_df)
    #print("\n\n\n")
    
    results['2020 MAE Value'] = metrics.mean_absolute_error(y_2020_actual_prices, y_2020_pred_prices)
    results['2020 MSE Value'] = metrics.mean_squared_error(y_2020_actual_prices, y_2020_pred_prices)
    results['2020 RMSE Value'] = np.sqrt(metrics.mean_squared_error(y_2020_actual_prices, y_2020_pred_prices))
    
    #Print the MAE, MSE and RMSE values
    #print(f"MAE, MSE and RMSE on 2020 Data for '{city}'")
    #print("*******************************************************************")
    #print('Mean Absolute Error:', results['2020 MAE Value'])
    #print('Mean Squared Error:',results['2020 MSE Value'] )
    #print('Root Mean Squared Error:', results['2020 RMSE Value'])
    #print("\n\n\n")
    
    # 2020 GRAPH
    ax4 = ml_2020_df.plot(figsize = (15,8))
    plt.legend(bbox_to_anchor=(0.5,0.5), loc='upper right', ncol=1)
    plt.xlabel("Month", fontsize = 16)
    plt.ylabel("Average Sale Price (USD)")
    plt.title(f"Average Sale Price for 2020: Actual Vs Predicted for city '{city}'")     
    plt.show()
    ax4.get_figure().savefig(f"{proj_root_dir}/images/{city}_2020_act_pred.png")
    
    #model.save(f"{parentdir}\models\naive_model_{city}.h5")
    pickle.dump(model, open(f"{proj_root_dir}/models/univariate_lr_model_{city}.pkl", 'wb'))
    return results,ml_df_test_set,ml_2020_df

# Call the functions to run the linear regression model for each city
cities = ['Miami-Fort Lauderdale','Lakeland','Los Angeles-Long Beach-Anaheim','San Jose']

metrics_all = []
ml_test_set_all = [] 
ml_2020_all = [] 
for city in cities:
    results,ml_df_test_set,ml_2020_df = lnr_model(df,city)
    metrics_all.append(results)
    ml_test_set_all.append(ml_df_test_set)
    ml_2020_all.append(ml_2020_df)

metrics_df = pd.DataFrame(metrics_all)   
print(metrics_df)
ml_test_set_df = pd.concat(ml_test_set_all) 
ml_2020_df = pd.concat(ml_2020_all)   
print(ml_2020_df.shape)

#Export the model data to CSV
output_path = f"{proj_root_dir}/data/processed"
#file_path_export=Path("/Housing_cleaned2.csv")
metrics_df.to_csv(Path(f"{output_path}/model_metrics.csv"))
ml_test_set_df.to_csv(Path(f"{output_path}/ml_test_set_act_pred.csv"))
ml_2020_df.to_csv(Path(f"{output_path}/ml_2020_act_pred.csv"))
