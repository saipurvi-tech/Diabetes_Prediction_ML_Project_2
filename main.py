#Importing the dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#Data Collection and Analysis [Diabetes Dataset - Females]
diabetes_data = pd.read_csv('diabetes.csv')
#print(diabetes_data.head()) 
#print(diabetes_data.shape) 
#print(diabetes_data.describe())
#print(diabetes_data['Outcome'].value_counts()) 
#print(diabetes_data.groupby('Outcome').mean()) 

#splitting the data and lables 
X = diabetes_data.drop(columns = 'Outcome')
Y = diabetes_data['Outcome']
#print(X)
#print(Y)

#Data standardilization 
# if there is difference in the range, then our model cannot predict it correctly. 
#so we try to standardize our data into a particular range 
#that will help our model to make better decisions. 
scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
standardized_data = scaler.transform(X)
#print(standardized_data)
X = standardized_data
#print(X)
#print(Y)

#Spliting the Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

#Training the Model 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

#Accuracy score and model evaluation
X_train_prediciton = classifier.predict(X_train)
training_dat_prediction = accuracy_score(X_train_prediciton, Y_train)
#print('Accuracy score of the training data: ', training_dat_prediction)
X_test_prediction = classifier.predict(X_test)
testing_data_prediction = accuracy_score(X_test_prediction, Y_test)
#print('Accuracy score of the testing data: ', testing_data_prediction)

#Making a predictive system
input_data = (5,166,72,19,175,25.8,0.587,51)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')