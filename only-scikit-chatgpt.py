import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import time

# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocessing
data.dropna(inplace=True)
data = pd.get_dummies(data) # Encode categorical variables
X = data.drop('breach', axis=1)
y = data['breach']

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
dbscan = DBSCAN(eps=1, min_samples=10).fit(X)

# Anomaly detection
isoforest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42).fit(X)
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Supervised learning
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)
svm = SVC(kernel='linear', random_state=42)

# Cross-validation
scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
scores_svm = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')

# Fit and predict
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
auc_roc = roc_auc_score(y_test, y_pred_rf)
training_time = time.time() - start_time

# Print results
print('Random Forest: Accuracy = {}, Precision = {}, Recall = {}, F1 Score = {}, AUC ROC = {}'.format(accuracy, precision, recall, f1, auc_roc))
print('Training time = {} seconds'.format(training_time))
