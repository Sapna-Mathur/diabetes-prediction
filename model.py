import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# reading and loading diabetes file
data = pd.read_csv('diabetes.csv')

# EDA
print("Columns Names : ", data.columns)   # printng name of columns in data
print("No. of rows and columns: ", data.shape)
# checking the number of zero values in data
print("No. of zero values in Glucose: ",data[data.Glucose == 0].shape[0])
print("No. of zero values in BMI: ",data[data.BMI == 0].shape[0])
print("No. of zero values in Age: ",data[data.Age == 0].shape[0])
print("No. of zero values in DiabetesPedigreeFunction: ",data[data.DiabetesPedigreeFunction == 0].shape[0])
print("No. of zero values in Insulin: ",data[data.Insulin == 0].shape[0])
print("No. of zero values in SkinThickness: ",data[data.SkinThickness == 0].shape[0])
print("No. of zero values in BloodPressure: ",data[data.BloodPressure == 0].shape[0])

# data preprocessing
filt = (data.BloodPressure != 0) & (data.Glucose != 0) & (data.BMI != 0)      #ignoring rows which have values zero
new_data = data[filt]
# print(new_data.shape)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = new_data[feature_names]
y = new_data.Outcome
# print(X.shape, y.shape)

# splitting new data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.shape, X_test.shape)

# Training
cls = LogisticRegression(max_iter=500, random_state=0)
cls.fit(X_train, y_train)
pickle.dump(cls, open('model.pkl', 'wb'))    #save the model in model.pkl file

# model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))
y_predict = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy Score :', accuracy)