import pandas as pd
import fbprophet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import mean

path = "D:\\Economia\\Series\\"
df = pd.read_csv(path + "cpi.csv", na_values=['.'], parse_dates=True, engine='c', dayfirst=True)
# df['CPI'] = df['CPI'])

df = df.fillna(method="ffill")

# Prophet requires columns ds (date) and y (value)
cpi = df.rename(columns={'DATE': 'ds', 'CPI': 'y'})

rows = cpi.shape[0]
split = rows - 2
cpi_train = cpi.loc[:split - 1, :] # Last item is inclusive for loc!!
cpi_test = cpi.loc[split:, :]


def try_errors(df):

    errors = {}

    for i in range(20):

        if i > 0:
            cpi_df = cpi[:-i * 2]
        else:
            cpi_df = cpi
        rows = cpi_df.shape[0]
        split = rows - 2
        cpi_train = cpi.loc[:split - 1, :]
        cpi_test = cpi.loc[split:, :]

        for x in [1E-3, 1E-2, 1E-1, 0.15]:

            x_exp = x

            #Find correct hyperparams
            cpi_prophet = fbprophet.Prophet(changepoint_prior_scale=x_exp)
            cpi_prophet.fit(cpi_train)

            # Make a future dataframe for 2  months
            cpi_forecast = cpi_prophet.make_future_dataframe(periods=cpi_test.shape[0], freq='M')

            # Make predictions
            cpi_forecast = cpi_prophet.predict(cpi_forecast)
            forecast_pts = cpi_forecast.loc[split:, :]

            error = mean_squared_error(cpi_test['y'], forecast_pts['yhat'])

            if not x in errors.keys():
                errors[x] = [error]
            else:
                errors[x].append(error)

        for key, value in errors.items():
            print(str(key) + " " + str(mean(value)))




try_errors(df)

"""
# Find correct hyperparams
cpi_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
cpi_prophet.fit(cpi_train)

# Make a future dataframe for 2 months
cpi_forecast = cpi_prophet.make_future_dataframe(periods=cpi_test.shape[0], freq='M')

# Make predictions
cpi_forecast = cpi_prophet.predict(cpi_forecast)
forecast_pts = cpi_forecast.loc[split:, :]
forecast_pts['y'] = cpi_test['y']

plt.plot(forecast_pts['ds'], forecast_pts['yhat'], forecast_pts['ds'], forecast_pts['y'])

# cpi_prophet.plot(cpi_forecast, xlabel='Date', ylabel='CPI')
plt.title('CPI Forecast')
plt.show()
"""
