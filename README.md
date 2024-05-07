# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Lubindher
RegisterNumber:  212222240056
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

df.head()

![ML21](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/2c54ae66-ff11-424d-8199-83bbe2fb7bfc)

df.tail()

![ML22](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/4a22d19e-6daf-4461-a4ca-9099c683d14c)

Array value of X

![ML23](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/9a96bf8b-5cc2-42cb-80c1-5f472fb282a1)

Array value of Y

![ML24](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/73e3d360-c5f6-42ff-99e5-9987bcc6f6df)

Values of Y prediction

![ML25](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/cb2885d2-6946-4d26-8920-351ff829809f)

Array values of Y test

![ML26](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/be2ad06a-54a2-4761-93bd-9fd8bf1e3d9c)

Training Set Graph

![ML27](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/97c9dc2e-d865-4509-a4ea-ba33f6a417c0)

Test Set Graph

![ML28](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/60b841ab-e317-4cf2-a3e4-55755840974a)

Values of MSE, MAE and RMSE

![ML29](https://github.com/22008496/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476113/62bafbdc-af6b-4e2f-a0d1-ed47389c45a1)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
