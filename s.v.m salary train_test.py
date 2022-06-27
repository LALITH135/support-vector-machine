

# import data

import pandas as pd
SalaryData_Test = pd.read_csv('C:\python notes\ASSIGNMENTS\S.V.M\SalaryData_Test(1).csv')
SalaryData_Train = pd.read_csv('C:\python notes\ASSIGNMENTS\S.V.M\SalaryData_Train(1).csv')

SalaryData_Test.shape
list(SalaryData_Test)
SalaryData_Test.describe
SalaryData_Test.head()
SalaryData_Test.corr()
SalaryData_Test.info()

# this data is in object.so, we have to convert into integers. 
object_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']

# label encoding
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in object_columns:
    SalaryData_Train[i] = number.fit_transform(SalaryData_Train[i])
    SalaryData_Test[i] = number.fit_transform(SalaryData_Test[i])

# spliting the data, by test size 0.25. 

from sklearn.model_selection import train_test_split
train,test = train_test_split(SalaryData_Test,test_size = 0.25,random_state=0)
test.head()
train_X = train.iloc[:,:13]
train_y = train.iloc[:,13]
list(train_X)
test_X  = test.iloc[:,:13]
test_y  = test.iloc[:,13]
list(test_X)

# STANDARDIZATION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

#support vector classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# kernel-linear
clf = SVC(kernel = "linear")
clf.fit(train_X,train_y)
pred_test_linear = clf.predict(test_X)
print("Accuracy :",accuracy_score(test_y, pred_test_linear).round(2))
# Accuracy : 0.81

# kernel-poly
clf2 = SVC(kernel = "poly")
clf2.fit(train_X,train_y)
pred_test_linear = clf2.predict(test_X)
print("Accuracy :",accuracy_score(test_y, pred_test_linear).round(2))
# Accuracy : 0.84

# kernel-rbf
clf3 = SVC(kernel = "rbf")
clf3.fit(train_X,train_y)
pred_test_linear = clf3.predict(test_X)
print("Accuracy :",accuracy_score(test_y, pred_test_linear).round(2))
# Accuracy : 0.85

# by checking each kernel accuracy,rbf is the best for these dataset.

#==========================================================================
