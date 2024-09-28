#!/usr/bin/env python
# coding: utf-8

# In[173]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import joblib 
import warnings
warnings.filterwarnings("ignore")


# In[181]:


df = pd.read_csv('default.csv')


# In[182]:


categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
df_cat = df[categorical_features]
df_cat['Defaulter'] = df['default']


# In[183]:


df_cat.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)


# In[186]:


for col in categorical_features:
  plt.figure(figsize=(10,10))
  fig, axes = plt.subplots(ncols=2,figsize=(13,10))
  df[col].value_counts().plot(kind="pie",ax = axes[0],subplots=True)
  sns.countplot(x = col, hue = 'Defaulter', data = df_cat)


# In[187]:


df.groupby('default')['AGE'].mean()


# In[188]:


df.SEX.value_counts()


# In[189]:


df['EDUCATION'].unique()


# In[190]:


df.EDUCATION.value_counts()


# In[191]:


df['EDUCATION'].replace([0, 6], 5, inplace=True)
df['EDUCATION'].replace(5, 4, inplace=True)
df.EDUCATION.value_counts()


# In[192]:


df.MARRIAGE.value_counts()


# In[193]:


df['MARRIAGE'].replace(0, 3, inplace=True)
df.MARRIAGE.value_counts()


# In[194]:


yes = df.default.sum()
no = len(df)-yes

# Percentage
yes_perc = round(yes/len(df)*100, 1)
no_perc = round(no/len(df)*100, 1)

import sys 
plt.figure(figsize=(7,4))
sns.set_context('notebook', font_scale=1.2)
sns.countplot('default',data=df, palette="Blues")
plt.annotate('Non-default: {}'.format(no), xy=(-0.3, 15000), xytext=(-0.3, 3000), size=12)
plt.annotate('Default: {}'.format(yes), xy=(0.7, 15000), xytext=(0.7, 3000), size=12)
plt.annotate(str(no_perc)+" %", xy=(-0.3, 15000), xytext=(-0.1, 8000), size=12)
plt.annotate(str(yes_perc)+" %", xy=(0.7, 15000), xytext=(0.9, 8000), size=12)
plt.title('COUNT OF CREDIT CARDS', size=14)
#Removing the frame
plt.box(False);


# In[195]:


le = LabelEncoder()

# Encode the 'Sex' column and insert it at the 3rd position
df.insert(2, 'Gender_encoded', le.fit_transform(df['SEX']))



# Print the modified DataFrame
print(df.head())


# In[196]:


le = LabelEncoder()

# Encode the 'Sex' column and insert it at the 3rd position
df.insert(3, 'education_encoded', le.fit_transform(df['EDUCATION']))


# In[197]:


df.insert(4, 'marriage_encoded', le.fit_transform(df['MARRIAGE']))


# In[198]:


df.head()


# In[199]:


df = df.drop('SEX', axis=1)
df = df.drop('EDUCATION', axis=1)
df = df.drop('MARRIAGE', axis=1)
df.head()


# In[200]:


df.drop('ID',axis=1, inplace=True)


# In[201]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[202]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[203]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[204]:


from sklearn.svm import SVC
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)
prediction = svc_model .predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print("Support Vector Machine: ",accuracy_score(prediction,y_test))
print(classification_report(prediction,y_test))
print(confusion_matrix(prediction,y_test))

from sklearn.linear_model import LogisticRegression
m = LogisticRegression()
m.fit(X_train,y_train)
pred = m.predict(X_test)
from sklearn.metrics import accuracy_score
print("Logistic Regression: ", accuracy_score(pred,y_test))
print(classification_report(pred,y_test))
print(confusion_matrix(pred,y_test))

from sklearn.tree import DecisionTreeClassifier
c = DecisionTreeClassifier(random_state=0)
c.fit(X_train,y_train)
y_tree = c.predict(X_test)
from sklearn.metrics import accuracy_score
print("Decision Tree: ", accuracy_score(y_tree,y_test))
print(classification_report(y_tree,y_test))
cm = (confusion_matrix(y_tree,y_test))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
y_ran = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("Random Forest: ", accuracy_score(y_ran,y_test))
print(classification_report(y_ran,y_test))
print(confusion_matrix(y_ran,y_test))


# In[205]:


from sklearn.tree import DecisionTreeClassifier
c = DecisionTreeClassifier(random_state=0)
c.fit(X_train,y_train)
y_tree = c.predict(X_test)
from sklearn.metrics import accuracy_score
print("Decision Tree: ", accuracy_score(y_tree,y_test))
print(classification_report(y_tree,y_test))
cm = (confusion_matrix(y_tree,y_test))


# In[206]:


labels = ['Not Defaulter', 'Defaulter']
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='d') #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[207]:


probs = c.predict_proba(X_test)

# Set new decision threshold for class 1
new_threshold = 0.5
y_pred = (probs[:, 1] >= new_threshold).astype(int)

# Evaluate the performance of your classifier with the new threshold
print(classification_report(y_test, y_pred))
c_dt = (confusion_matrix(y_pred,y_test))


# In[208]:


labels = ['Not Defaulter', 'Defaulter']
ax= plt.subplot()
sns.heatmap(c_dt, annot=True, ax = ax, fmt='d') #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[209]:


from sklearn.svm import SVC
svc_model = SVC(kernel='rbf',probability=True)
svc_model.fit(X_train, y_train)
prediction = svc_model .predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print("Support Vector Machine: ",accuracy_score(prediction,y_test))
print(classification_report(prediction,y_test))
cf = (confusion_matrix(prediction,y_test))


# In[210]:


labels = ['Not Defaulter', 'Defaulter']
ax= plt.subplot()
sns.heatmap(cf, annot=True, ax = ax, fmt='d') #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[211]:



# Get the decision scores for the test data
scores = svc_model.decision_function(X_test)

# Set the new decision threshold
new_threshold = -0.15

# Apply the new decision threshold to the scores
y_pred = (scores >= new_threshold).astype(int)

# Evaluate the performance with the new threshold
print(classification_report(y_test, y_pred))
c_chang = (confusion_matrix(y_pred,y_test))


# In[212]:


labels = ['Not Defaulter', 'Defaulter']
ax= plt.subplot()
sns.heatmap(c_chang, annot=True, ax = ax, fmt='d') #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[213]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
y_ran = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("Random Forest: ", accuracy_score(y_ran,y_test))
print(classification_report(y_ran,y_test))
c_ran = (confusion_matrix(y_ran,y_test))


# In[214]:


labels = ['Not Defaulter', 'Defaulter']
ax= plt.subplot()
sns.heatmap(c_ran, annot=True, ax = ax, fmt='d') #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[215]:


d = pd.read_csv('test.csv')
d['EDUCATION'].replace([0, 6], 5, inplace=True)
d['EDUCATION'].replace(5, 4, inplace=True)

d['MARRIAGE'].replace(0, 3, inplace=True)

le = LabelEncoder()
# Encode the 'Sex' column and insert it at the 3rd position
d.insert(2, 'Gender_encoded', le.fit_transform(d['SEX']))
# Encode the 'Sex' column and insert it at the 3rd position
d.insert(3, 'education_encoded', le.fit_transform(d['EDUCATION']))
d.insert(4, 'marriage_encoded', le.fit_transform(d['MARRIAGE']))
d.head()


# In[216]:



d = d.drop('MARRIAGE', axis=1)
d.head()


# In[217]:


d = d.drop('SEX', axis=1)
d = d.drop('EDUCATION', axis=1)


# In[218]:


d.drop('ID',axis=1, inplace=True)


# In[219]:


d.head()


# In[220]:


X_1 = d.iloc[:,:-1].values
y_1 = d.iloc[:,-1].values


# In[221]:


scaled_new_data = scaler.transform(X_1)


# In[222]:


d = c.predict(scaled_new_data)
rf = clf.predict(scaled_new_data)
log = m.predict(scaled_new_data)
s = svc_model.predict(scaled_new_data)


# In[223]:


print(d)
print( log)
print(rf)
print(s)
#print(new_pred)
print(y)


# In[224]:


new = [[125000,1,2,1,35,2,1,1,1,0,0,12000,0,188,7000,4000,2000,0,0,0,0,0,0]]


# In[225]:


scaled_new_data = scaler.transform(new)


# In[226]:


d = c.predict(scaled_new_data)
rf = clf.predict(scaled_new_data)
log = m.predict(scaled_new_data)
s = svc_model.predict(scaled_new_data)
print(d)
print( log)
print(rf)
print(s)


# In[227]:


joblib.dump(svc_model, 'mymodel.joblib')
joblib.dump(clf, 'rf_model.joblib')
joblib.dump(d, 'dt_model.joblib')
joblib.dump(m, 'log_model.joblib')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




