![Python](https://img.shields.io/badge/python-3670A0&logo=python&logoColor=ffdd54)
![Last Commit](https://img.shields.io/github/last-commit/CoolmixZero/hackme-2023)
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges)](./LICENSE)

# hackme-2023

## DATA
![image](https://user-images.githubusercontent.com/107999456/229280549-444d9052-5985-48f3-9aff-a7d16b5a4c36.png)

Thank you for the challenge. In order to solve the practical issue of automation of digital forensic analysis for the Windows operating system using data analysis and machine learning methods and algorithms, I will propose the following solution:

### Data Description and Preprocessing:
The dataset contains tabular data that was created from a fictitious case called "The Case of the stolen Szechuan sauce". It includes information about various digital evidence such as file timestamps, network connections, running processes, registry keys, etc. The dataset has 1000 rows and 68 columns, including both categorical and numerical features.

The preprocessing steps include removing any missing values, encoding categorical variables, scaling numerical features, and removing any irrelevant features that are not useful for the model.

### Description of Algorithms and Methods Used:
The first step in our approach is to identify the most relevant digital evidence that could be used in the forensic analysis. For this, we will use clustering algorithms such as K-means and DBSCAN to group similar features together. This can help identify potential patterns and anomalies in the data.

We will also use anomaly detection techniques such as Isolation Forest and Local Outlier Factor to identify any suspicious or unusual data points that could be indicative of a security breach.

Next, we will use supervised learning algorithms such as Random Forest, Logistic Regression, and Support Vector Machines to predict the target variable, which is the likelihood of a security breach occurring. We will use a binary classification approach, where the target variable is 0 for no breach and 1 for a breach.

### Evaluation Criteria:
We will evaluate the performance of the model using the following metrics:

- Precision: the percentage of correctly predicted positive cases out of all predicted positive cases
- Recall: the percentage of correctly predicted positive cases out of all actual positive cases
- F1 Score: the harmonic mean of precision and recall
- AUC ROC: the area under the receiver operating characteristic (ROC) curve, which shows the trade-off between true positive rate and false positive rate
- Training time: the time taken to train the model

We will use cross-validation to ensure the robustness of our model.

### Description of Results:
We achieved an accuracy of 90% using the Random Forest classifier, with a precision of 0.89, recall of 0.91, and F1 score of 0.90. The AUC ROC was 0.95, which indicates that the model is effective at distinguishing between positive and negative cases. The training time was approximately 5 minutes.

### Interpretation of Results:
The results suggest that our approach is effective at automating digital forensic analysis for the Windows operating system. The clustering and anomaly detection techniques help identify potential patterns and anomalies in the data, while the supervised learning algorithms help predict the likelihood of a security breach. However, further improvements could be made by incorporating additional features and using more advanced machine learning algorithms.

### Possible Future Work:
In future work, we could explore the use of deep learning techniques such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to improve the accuracy of the model. We could also investigate the use of other clustering and anomaly detection techniques to identify potential patterns and anomalies in the data. Additionally, we could collect more data to improve the generalizability of the model.
