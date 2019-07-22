import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x=pd.read_csv("http://www.stat.ufl.edu/~winner/data/airq402.dat",names=['CITY1','CITY2','Total Avg Fare','Distance','Avg wkly pssngrs','Lead airline','lead market share','Lead Average fare','Low price airline','low market share','price'],sep="\s+")
#print(x.skew())
x=x[['Total Avg Fare','Distance','Avg wkly pssngrs','lead market share','Lead Average fare','low market share','price']]
y= x['Total Avg Fare'].values
x1 = x['Lead Average fare'].values
plt.subplot(1,2,1)
plt.scatter(x1,y,color='green')
plt.xlabel("LEAD AIRLINE AVG FARE")
plt.ylabel("TOTAL AVG FARE")
plt.title("LEAD AIRLINE AVG FARE VS TOTAL AVG FARE ")

reg=linear_model.LinearRegression()
m=len(x1)
x1=x1.reshape((m,1))
z=reg.fit(x1,y)
plt.subplot(1,2,2)
plt.plot(x1,reg.predict(x1),color='red')
plt.xlabel("LEAD AIRLINE AVG FARE")
plt.ylabel("TOTAL AVG FARE")
plt.title("LR MODEL FOR LEAD AIRLINE AVG FARE VS TOTAL AVG FARE ")
plt.show()
x3_train,x3_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=1)
print("ACCURACY FOR LEAD AIRLINE AVG FARE VS TOTAL AVG FARE=",reg.score(x3_test,y_test))


x1 = x['Distance'].values
plt.subplot(1,2,1)
plt.scatter(x1,y,color='green')
plt.xlabel("DISTANCE")
plt.ylabel("TOTAL AVG FARE")
plt.title("DISTANCE VS TOTAL AVG FARE ")

reg=linear_model.LinearRegression()
m=len(x1)
x1=x1.reshape((m,1))
z=reg.fit(x1,y)
plt.subplot(1,2,2)
plt.plot(x1,reg.predict(x1),color='red')
plt.xlabel("DISTANCE")
plt.ylabel("TOTAL AVG FARE")
plt.title("DISTANCE AVG FARE VS TOTAL AVG FARE ")
plt.show()
x3_train,x3_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=1)
print("ACCURACY FOR DISTANCE VS TOTAL AVG FARE=",reg.score(x3_test,y_test))



x1 = x['Avg wkly pssngrs'].values
plt.subplot(1,2,1)
plt.scatter(x1,y,color='magenta')
plt.xlabel("Avg wkly pssngrs")
plt.ylabel("TOTAL AVG FARE")
plt.title("Avg wkly pssngrs VS TOTAL AVG FARE ")

reg=linear_model.LinearRegression()
m=len(x1)
x1=x1.reshape((m,1))
z=reg.fit(x1,y)
plt.subplot(1,2,2)
plt.plot(x1,reg.predict(x1),color='yellow')
plt.xlabel("Avg wkly pssngrs")
plt.ylabel("TOTAL AVG FARE")
plt.title("Avg wkly pssngrs VS TOTAL AVG FARE ")
plt.show()
x3_train,x3_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=1)
print("ACCURACY FOR Avg wkly pssngrs VS TOTAL AVG FARE=",reg.score(x3_test,y_test))



x1 = x['lead market share'].values
plt.subplot(1,2,1)
plt.scatter(x1,y,color='black')
plt.xlabel("lead market share")
plt.ylabel("TOTAL AVG FARE")
plt.title("lead market share VS TOTAL AVG FARE ")

reg=linear_model.LinearRegression()
m=len(x1)
x1=x1.reshape((m,1))
z=reg.fit(x1,y)
plt.subplot(1,2,2)
plt.plot(x1,reg.predict(x1),color='blue')
plt.xlabel("lead market share")
plt.ylabel("TOTAL AVG FARE")
plt.title("lead market share VS TOTAL AVG FARE ")
plt.show()
x3_train,x3_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=1)
print("ACCURACY FOR lead market share VS TOTAL AVG FARE=",reg.score(x3_test,y_test))



x1 = x['low market share'].values
plt.subplot(1,2,1)
plt.scatter(x1,y,color='cyan')
plt.xlabel('low market share')
plt.ylabel("TOTAL AVG FARE")
plt.title("low market share VS TOTAL AVG FARE ")

reg=linear_model.LinearRegression()
m=len(x1)
x1=x1.reshape((m,1))
z=reg.fit(x1,y)
plt.subplot(1,2,2)
plt.plot(x1,reg.predict(x1),color='magenta')
plt.xlabel("low market share")
plt.ylabel("TOTAL AVG FARE")
plt.title("low market share VS TOTAL AVG FARE ")
plt.show()
x3_train,x3_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=1)
print("ACCURACY FOR low market share VS TOTAL AVG FARE=",reg.score(x3_test,y_test))



x1 = x['price'].values
plt.subplot(1,2,1)
plt.scatter(x1,y,color='black')
plt.xlabel("PRICE")
plt.ylabel("TOTAL AVG FARE")
plt.title("PRICE VS TOTAL AVG FARE ")

reg=linear_model.LinearRegression()
m=len(x1)
x1=x1.reshape((m,1))
z=reg.fit(x1,y)
plt.subplot(1,2,2)
plt.plot(x1,reg.predict(x1),color='yellow')
plt.xlabel("PRICE")
plt.ylabel("TOTAL AVG FARE")
plt.title("PRICE VS TOTAL AVG FARE ")
plt.show()
x3_train,x3_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=1)
print("ACCURACY FOR PRICE VS TOTAL AVG FARE=",reg.score(x3_test,y_test))
