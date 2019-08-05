import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df=pd.read_csv('Customer_data.csv')
X=df.iloc[:,3:13].values
y=df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
labelencoder2=LabelEncoder()
X[:,2]=labelencoder2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

sc_X=StandardScaler()
X=sc_X.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


#import keras
#Sequential module is used to initialize the neural network
from keras.models import Sequential
'''
Dense module is used for building the layers of our neural network.
Dense module also takes care of the random initialization of our weights.
'''

from keras.layers import Dense
classifier=Sequential()
 

'''
add method is used to add the hidden layers in the neural network. 
Dense module helps us to initialize the properties of the hidden layer. 
init specifies how the values of the weights are to be initialized. 
activation= Activation function. relu= Rectifier.
input_dim=no. of independent variables.
output_dim=no. of units/nodes in the hidden layer. Decided by a genral formula [(x+y)/2] where x and y are the number of nodes in the input and output layers respectively.
'''

#Below is the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#Below is the second hidden layer
#Didn't use the input_dim parameter as it is the second hidden layer. 

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#Blow is the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compile method will apply a stochastic gradient descent on the whole neural network.
'''
optimizer specifies the type of stochastic gradient descent to be used. Gives the optimal weights.

metrics specifies the type of metrics used to improve the output of the network.
'''
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

'''
batch_size specifies the no. of training examples after which the weights are to be updated.
nb_epoch specifies the number of epochs
'''

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

cm=confusion_matrix(y_test,y_pred)


