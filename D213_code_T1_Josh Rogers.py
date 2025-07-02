import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from math import sqrt
import warnings
from fpdf import FPDF

# Create a DataFrame from the data
data = pd.read_csv('teleco_time_series .csv')
data.set_index('Day', inplace=True)
data = data.dropna()
print(data.head(5))
print(data.shape)

# Saving the cleaned dataset
cleaned_data_path = 'cleaned_revenue_data.csv'
data.to_csv(cleaned_data_path, index=False)

# Plotting the time series
plt.figure(figsize=(12, 5))
plt.plot(data['Revenue'], label='Revenue over Time', color='blue')
plt.title('Revenue Time Series')
plt.xlabel('Day')
plt.ylabel('Revenue (Million Dollars)')
plt.grid()
plt.legend()
plt.show()

# Splitting the data
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
print(f"Train set size: {train.shape}, Test set size: {test.shape}")

# Rolling Mean and STD
rolmean = data['Revenue'].rolling(window=50).mean()
rolstd = data['Revenue'].rolling(window=50).std()
print(rolmean, rolstd)

orig = plt.plot(data['Revenue'], color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling STD')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.show()

# Plot ACF before differencing
plt.subplot(1, 2, 1)
plot_acf(data['Revenue'], lags=40, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

# Plot PACF before differencing
plt.subplot(1, 2, 2)
plot_pacf(data['Revenue'], lags=40, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# Decompose the time series
result = seasonal_decompose(data['Revenue'], model='additive', period=365)
result.plot()
plt.show()

#Checking Seasonality using STL
stl_model = STL(data['Revenue'], period=365)
res = stl_model.fit()
fig = res.plot()

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(data['Revenue'], autolag='AIC')
print("1. ADF Statistic:", adf_test[0])
print("2. p-value:", adf_test[1])
print("3. #Lags Used:", adf_test[2])
print("4. #Observations Used:", adf_test[3])
print("5. #Critical Values Used:")
for key, value in adf_test[4].items():
  print("\t", key, ":", value)

# Check for stationarity
if adf_test[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is non-stationary.")

#Differencing
data['Revenue_diff'] = data['Revenue'].diff()
data = data.dropna(subset=['Revenue_diff'])

# Perform Augmented Dickey-Fuller test on the differenced time series
adf_test2 = adfuller(data['Revenue_diff'].dropna(), autolag='AIC')
print("1. ADF Statistic:", adf_test2[0])
print("2. p-value:", adf_test2[1])
print("3. #Lags Used:", adf_test2[2])
print("4. #Observations Used:", adf_test2[3])
print("5. #Critical Values Used:")
for key, value in adf_test2[4].items():
  print("\t", key, ":", value)

# Check for stationarity
if adf_test2[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is non-stationary.")

# Calculate and plot ACF and PACF
plt.figure(figsize=(12, 6))

# Plot ACF after differencing
plt.subplot(1, 2, 1)
plot_acf(data['Revenue_diff'], lags=40, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

# Plot PACF after differencing
plt.subplot(1, 2, 2)
plot_pacf(data['Revenue_diff'], lags=40, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

result = seasonal_decompose(data['Revenue_diff'].dropna(), model='additive', period=365)
result.plot()
plt.show()

# Apply AutoARIMA
warnings.filterwarnings('ignore')
stepwise_fit = auto_arima(data['Revenue'], trace=True, suppress_warnings=True)
# Print the summary of the model
print(stepwise_fit.summary())

# Fit the ARIMA model based on manually determined p, d, q
manual_model = ARIMA(train['Revenue'], order=(1, 2, 1))  
manual_model = manual_model.fit()
residuals = train['Revenue'] - manual_model.fittedvalues
RSS = sum(residuals**2)
print('RSS: %.4f' % RSS)
print("Manual ARIMA Model Summary on training data:")
print(manual_model.summary())

# Fit the ARIMA model based on suggested p, d, q
auto_model = ARIMA(train['Revenue'], order=(1, 1, 0))  
auto_model = auto_model.fit()
residuals = train['Revenue'] - auto_model.fittedvalues
RSS = sum(residuals**2)
print('RSS: %.4f' % RSS)
print("AutoARIMA Model Summary on training data:")
print(auto_model.summary())

fig, ax = plt.subplots()
data.plot(ax=ax)
plot_predict(manual_model, start=len(train), end=len(train)+len(test)-1, ax=ax)
plt.show()

residuals = manual_model.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()

acf_res = plot_acf(residuals, lags=40)
pacf_res = plot_pacf(residuals, lags=40)

# Make predictions on the test set
start = len(train)
end = len(train) + len(test) - 1
pred = manual_model.predict(start=start, end=end, typ='levels')
print(pred)
pred.index = test.index
plt.figure(figsize=(10,6))
pred.plot(legend=True, label='Predicted')
test['Revenue'].plot(legend=True, label='Actual')
plt.title('ARIMA Model Predictions vs Actual Revenue')
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.show()

# Calculate test mean and RMSE
print("Test Mean:", test['Revenue'].mean())

rmse = sqrt(mean_squared_error(test['Revenue'], pred))
print(f"RMSE: {rmse}")

forecast_index = np.arange(732, 762, 1)
print(forecast_index)
pred = manual_model.predict(start=len(data), end=len(data) + 29, typ='levels').rename('ARIMA Predictions')
pred.index = forecast_index
print('Forecasted Revenue:')
print(pred)

prediction = manual_model.get_prediction(start=len(data), end=len(data) + 29)
print(prediction)

# Get the prediction intervals (default 95% confidence level)
conf_int = prediction.conf_int(alpha=0.05)

# Print the prediction intervals for the forecast
print("Prediction Interval (95% Confidence):")
print(conf_int)

# Plot the actual vs predicted data
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Revenue'], label='Test Revenue', color='blue') 
plt.plot(pred.index, pred, label='Predicted Revenue', color='red', linestyle='--')  
plt.legend(loc='best')
plt.title('ARIMA Model: Actual vs Predicted Revenue')
plt.xlabel('Day')
plt.ylabel('Revenue')
plt.show()


# Create a PDF object (Jam75, 2023)
pdf = FPDF()
pdf.add_page()
pdf.add_font('NotoSans', '', 'NotoSans-VariableFont_wdth,wght.ttf', uni=True)
pdf.set_font("NotoSans", size=12)
pdf.cell(200, 10, txt="Forecasting Report", ln=True, align='C')

#Text
text = """
Results:

Both AutoARIMA and manually created ARIMA models were considered for this analysis. While the AutoARIMA provided the p, d, q values of 1, 1, 0, respectively, the resulting ACF and PACF visualizations suggested that additional differencing may be required. Thus, the p, d, q values of 1, 2, 1, were selected for modeling. These values were vetted by reviewing the SARIMAX results and reviewing the p-values to ensure they fell into the appropriate 95% significance level.

The prediction intervals are significantly large. The wide range of these intervals suggests a relatively high level of uncertainty with the model. This is especially important to consider when determining potential bonuses to employees as these numbers represent millions of dollars. There may be higher risk involved based on the significant width of the forecasted intervals.

The length of the forecast was determined using a couple of different factors. First, the shorter 30 day forecast length is usually going to yield more accurate model results. This is because the model will rely much more on recent day considerations. Secondly, the question being posed involves holiday bonuses which is more current and does not require major long-term projections.
The model was evaluated using both the RMSE and a visualization of the residual data. The RMSE value was determined to be 2.29. Given the large range of the revenue data this value appears validate the accuracy of the model. However, if we consider that this value’s units is millions of dollars, it may be too large of a number to put too much faith in the accuracy of the model.

Recomendations:

As we consider the original question of a predictive time series model and its ability to provide guidance for a company in relation to offering holiday bonuses, the resulting model is helpful. The original time series does show an overall upward trend in revenue which is encouraging, and the model developed further supports an upward trend over the next 30 days. However, the variability of the model demonstrates a lack of confidence of the predicted values. Even given this information, my recommendation for the organization is to still consider providing employee bonuses with the caveat to ensure that the amounts are reasonable and account for potential revenue fluctuation. Providing such a gift to the employees will improve morale and will hopefully reflect in organizational performance moving forward. Future considerations surrounding holiday bonuses and their impact on revenue can be evaluated once enough such data is collected.
"""

# Add the text to the PDF
pdf.multi_cell(0, 10, txt=text)

# Save the PDF to a file
pdf.output("Revenue_Forecast_Results.pdf")








