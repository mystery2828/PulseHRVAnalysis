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
df = pd.read_excel('HRV.xlsx')

# Convert the classes into 'B','C','M'
bcm = df.iloc[:,0]

# Seperate the patient ID from dataframe
df = df.iloc[:,1:]

# Fill all the NaN values
df['STD_NN'] = df['STD_NN'].fillna(df['STD_NN'].mean())

# Rename the classes
for i in range(len(df)):
    df.iloc[i,-1] = bcm[i][0]

# Split the classes from the dataset and assign it to a variable
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

plt.figure()
plt.grid(axis='both')

# Features in the dataset
col = ['meanNN', 'STD_NN', 'HR', 'Lfnu', 'Hfnu', 'LF/HF', 'APEN', 'CD', 'PTT',
       'PTT_SD']

# Synthetic Minority oversampling technique to balance the dataset
oversample = SMOTE()
X,y = oversample.fit_resample(X,y)

# Feature Selection
feature_scores = SelectKBest(score_func=chi2,k=10)
fit = feature_scores.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcol = pd.DataFrame(X.columns)
visual = pd.concat([dfcol,dfscores],axis=1)
visual.columns = ['Specs','Score']

# Applying the feature values to the features 
for i in range(len(list(fit.scores_/np.mean(df[X.columns])))):
    X.iloc[:,i]*=(fit.scores_[i]/max(fit.scores_))

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
classifier = SVC(kernel='rbf',gamma='auto',probability=True,decision_function_shape='ovo',verbose=True)
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
plt.title('HRV model 3 class')
plt.plot(y_pred)
plt.plot(yt)
plt.show()

# Construct a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# End of model
print('End of model')