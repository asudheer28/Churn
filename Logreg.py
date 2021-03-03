

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('PROJECT2.csv')

df1 = data.copy()
df1.drop(['Unnamed: 0'], axis = 1,inplace=True)

X = df1.iloc[:,:3]
y= df1["Status_ACTIVE"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=123)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import pickle

pickle.dump(logreg, open("logreg.pkl", "wb"))
logreg = pickle.load(open('logreg.pkl','rb'))

print(logreg.predict([[25,50000,12]]))








