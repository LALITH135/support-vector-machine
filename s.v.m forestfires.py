
# importing the datasets.

import pandas as pd

df = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\S.V.M\\forestfires.csv')
df.shape
list(df)
df.describe
df.head()
df.corr()
df.info()

# label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['month'] = LE.fit_transform(df['month'])
df['month'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['day'] = LE.fit_transform(df['day'])
df['day'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['size_category'] = LE.fit_transform(df['size_category'])
df['size_category'].value_counts()

# plotting in boxplot
import seaborn as sns

sns.boxplot(x="day",y="size_category",data=df,palette = "hls")
sns.boxplot(x="month",y="size_category",data=df,palette = "hls")


# split x&y variables

X = df.iloc[:,0:29]
y = df.iloc[:,30]
list(X)

# splitting train& test

from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Loading SVC 
# Training a classifier - kernel='rbf','poly','linear'
from sklearn.svm import SVC
SVC()
# kernel's is in comment checking each by removing comment.

#clf = SVC(kernel='linear')
#clf = SVC(kernel='poly') 
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
y_pred_train=clf.predict(X_train)

# import the metrics class
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# checking each kernel accuracy.
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred).round(2))
print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
cm = metrics.confusion_matrix(y_train, y_pred_train)
print(cm)

#poly

#Testing Accuracy: 0.78      
#Training Accuracy : 0.76    
#------------------------------
#linear

#Testing Accuracy: 0.98
#Training Accuracy : 1.0
#-----------------------
#rbf

#Testing Accuracy: 0.74
#Training Accuracy : 0.75

# by checking each kernel accuracy,linear is the best for these dataset.

#========================================================================






















