###Vedant Mukhedkar
###TETA75
###Linear Regression 

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# loading dataset
data = pd.read_csv(r'/content/drive/My Drive/Colab Notebooks/Fish.csv')
print(data, "\n")

data.shape

data.head()

#Dropping the unnecessary columns
data.drop(columns=['Species', 'Length1','Length2','Length3','Height'], inplace = True)\

data.head()

#plot using seaborn library
sns.catplot(x="Weight",y="Width",data=data)

sns.relplot(x="Weight",y="Width",data=data,kind="line")

#Setting the value for X and Y
x = data[['Weight']]
y = data['Width']

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
#70% for train and 30% for test

#Implementing the linear model
#Fitting the Linear Regression model
from sklearn.linear_model import LinearRegression
slr = LinearRegression()  
slr.fit(x_train, y_train)


#Intercept and Coefficient
#y=mx+c (c = coefficient and m = intercept)
print("Intercept: ", slr.intercept_)
print("Coefficient: ", slr.coef_)

#Prediction of test set
y_pred_slr= slr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_slr))

#Actual value and the predicted value
slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
slr_diff.head()


#Line of best fit
plt.scatter(x_test,y_test)
plt.plot(x_test, y_pred_slr,'r')
plt.xlabel('Weight')
plt.ylabel('Width')
plt.title('Actual vs Predicted')
plt.show()


#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))
print('R squared: {:.2f}'.format(slr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
