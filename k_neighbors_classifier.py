# data from https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
# Occupancy Detection Data Set
# Time-serie multivariate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from prepare_data import scale

# import data
df = pd.read_csv('datatraining.txt', sep=",")
# remove irrelevant feature
df.drop(columns=['date'], inplace=True)
# define target feature
target = "Occupancy"
# prepare the dataframe
df = scale(df=df, ordinal_features=None, target=target)
# define x and y
X, y = df.drop(columns=target), df[target]
# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# define the model
knn = KNeighborsClassifier(n_neighbors=3)
# fit to train data
knn = knn.fit(X_train, y_train)
# predict test data
y_pred = knn.predict(X_test)
# Precision, recall, f-score from the multi-class support function
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred), 2))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
disp.plot(cmap=plt.cm.Blues)
plt.show()