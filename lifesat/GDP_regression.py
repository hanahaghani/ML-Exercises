import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#prepare the data
lifesat=pd.read_csv("lifesat.csv")
x=lifesat[["GDP per capita (USD)"]].values
y=lifesat[["Life satisfaction"]].values

#visualize the data
lifesat.plot(kind='scatter',grid=True,x="GDP per capita (USD)",y="Life satisfaction")
plt.axis([23_500,62_500,4,9])
plt.show()

#select a linear model
model=LinearRegression()

#train the model
model.fit(x,y)

#make a prediction
x_new=[[37_655.2]]#GDP per capita in 2020
print(model.predict(x_new))