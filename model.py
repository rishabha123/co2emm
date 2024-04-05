import os
import urllib.request as url
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
saved_path = 'ENGINE_DATA.csv'
url.urlretrieve(path, saved_path)
new_data = pd.read_csv(saved_path)
cdf = new_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
axs[0, 0].set_xlabel("FUELCONSUMPTION_COMB")
axs[0, 0].set_ylabel("Emission")

axs[0, 1].scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
axs[0, 1].set_xlabel("CYLINDER")
axs[0, 1].set_ylabel("Emission")

axs[1, 0].scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
axs[1, 0].set_xlabel("Engine size")
axs[1, 0].set_ylabel("Emission")

mask = np.random.rand(len(cdf)) < 0.8
train = cdf[mask]
test = cdf[~mask]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Plotting the regression line using the training data
axs[1, 0].scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
axs[1, 0].set_xlabel("Engine size")
axs[1, 0].set_ylabel("Emission")

# Plotting the regression line
train_predictions = regr.predict(train_x)
axs[1, 1].plot(train.FUELCONSUMPTION_COMB, train_predictions, '-r')
axs[1, 1].set_xlabel("Fuel consumption")
axs[1, 1].set_ylabel("Emission")
#plt.show()

pickle.dump(regr,open("model.pkl","wb"))
