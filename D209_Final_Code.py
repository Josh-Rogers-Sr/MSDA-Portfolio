import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

# Import churn_clean.csv specifically bringing in only the columns considered for this analysis
customerdata = pd.read_csv('churn_clean.csv',usecols=['Children','Age','Income','Gender','Contract','PaymentMethod','Tenure','MonthlyCharge', 'Churn'])

# Create a new object that removes null rows
filtered_customerdata = customerdata.dropna()

# Create a new data frame in preparation for one hot encoding
customerdata_ohe = filtered_customerdata 

# Select categorical variables and store them in a list
categorical_cols = ['Churn', 'PaymentMethod','Contract','Gender']

# Create an object that passes in the categorical variables and the data frame with new columns that have been one hot encoded with integers
col_ohe = pd.get_dummies(filtered_customerdata[categorical_cols], drop_first=False, dtype=int)

# Creates a new object that concatenates the one hot encoded columns and removes the original columns and Churn
customerdata_ohe = pd.concat((customerdata_ohe, col_ohe), axis=1).drop(['Churn','PaymentMethod','Contract','Gender', 'Churn_No'], axis=1)

# Creates a dictionary with the column name and the new column name
#https://www.geeksforgeeks.org/how-to-create-a-dictionary-in-python/
dict = {'Churn_Yes':'Churn'} 

# Renames the columns in the data frame
customerdata_ohe.rename(columns=dict, inplace=True) 

correlation = customerdata_ohe.corr()
sns.heatmap(correlation, cmap="RdYlGn", annot=True)

feature_names = ['MonthlyCharge', 'Churn','Children','Age','Income','Gender_Male', 'Gender_Female', 'Gender_Nonbinary','Contract_Month-to-month','PaymentMethod_Bank Transfer(automatic)', 'PaymentMethod_Credit Card (automatic)', 'PaymentMethod_Electronic Check', 'PaymentMethod_Mailed Check', 'Contract_One year', 'Contract_Two Year']

sns.histplot(data=customerdata_ohe, x='Tenure', binwidth=10, label='Tenure')

# Prepares the data for the Decision Tree
X = customerdata_ohe[feature_names].values
y = customerdata_ohe['Tenure'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)

dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=0.1, random_state=12)

dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

#Visualize the decision tree
plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=feature_names, filled=True)

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})

#Display actual vs. predicted tenure values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Tenure')
plt.ylabel('Predicted Tenure')
plt.title('Actual vs Predicted Tenure (Months)')

#Display residual data between actual and predicted tenure values
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Tenure')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

#Metrics to evaluate algorithm performance
#(Decision trees in Python with scikit-learn, 2023)
print('R-Squared:', metrics.r2_score(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Convert split results to Pandas data frames
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

#Exports the data frames to csv files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)