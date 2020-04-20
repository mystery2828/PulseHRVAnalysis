# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:23:36 2020

@author: Akash
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:39:21 2020

@author: Akash
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.feature_selection import SelectKBest, chi2

plt.rcParams['axes.grid'] = True

# Read the dataset as a dataframe
df = pd.read_excel('pulse.xlsx')

# Seperate the patient ID from dataframe
df = df.iloc[:,1:]

# Fill all the NaN values
df['PHR'] = df['PHR'].fillna(df['PHR'].mean())

# Split the classes from the dataset and assign it to a variable
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

plt.figure()
plt.grid(axis='both')

# Features in the dataset
col = ['PR', 'PP', 'PT1', 'PT2', 'PT3', 'PDID', 'PHR', 'SVA', 'STRESS',
       'RATIO', 'Unnamed: 11']

# Synthetic Minority oversampling technique to balance the dataset
oversample = SMOTE()
X,y = oversample.fit_resample(X,y)

# Split the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, train_size = 0.8, random_state = 5)

# Normalize the values for better prediction efficiency
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Develop a SVM classifier model to train the data Use onevsone or onevsrest model
from sklearn.svm import SVC
classifier = SVC(decision_function_shape='ovo')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Converting the object into array
yt = []
for ele in y_test:
    yt.append(ele)

# Calculate the efficiency
count = 0
for i in range(len(yt)):
    if y_pred[i] == yt[i]:
        count+=1
eff = count/len(yt)
print('Efficiency: {}'.format(eff))
print('F1 score: {}'.format(f1_score(yt, y_pred, average='macro')))
print('Precision score: {}'.format(precision_score(yt, y_pred, average='macro')))
print('Jaccard score: {}'.format(jaccard_score(yt, y_pred, average=None)))

# Plot the efficiency data onto a graph
plt.title('pulse model 5 class')
plt.plot(y_pred)
plt.plot(yt)
plt.show()

# Construct a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# End of model
print('End of model')