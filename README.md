Breast Cancer Diagnosis Prediction 🩺🔬

📋 Project Description

This project aims to build, train, and compare several machine learning models to predict whether a breast cancer tumor is benign or malignant based on features extracted from digitized images of a fine needle aspirate (FNA) of a breast mass.

The dataset used is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

📈 Models Used :Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes, Bagging, AdaBoost (Boosting), Stacking Classifier

⚙️ Libraries Required: numpy, pandas, seaborn, matplotlib, scikit-learn, imbalanced-learn (for SMOTE), logging

🧪 Workflow

Load Data from wdbc.data

Preprocessing : Drop ID + Encode diagnosis (M → 1, B → 0) + Check missing values & duplicates

Exploratory Data Analysis (EDA) : Class distribution plot + Feature correlation heatmap

Data Splitting : 80% train, 20% test (stratified)

Standardization : Scale features using StandardScaler

Model Training : Train and evaluate multiple ML models

Model Evaluation : Confusion Matrix, Classification Report, ROC Curves, Accuracy comparison, Feature Importance (Random Forest)

📊 Visualizations Included

Class distribution, Correlation heatmap, Confusion matrix for each model ROC curve comparison, Barplot of model accuracies, Random Forest feature importance

🎯 Project Goals

Identify the best machine learning model for breast cancer diagnosis.

Improve model performance using:

SMOTE (oversampling)

Feature Scaling

Ensemble methods (Bagging, Boosting, Stacking)


🙌 Acknowledgements
Dataset provided by the UCI Machine Learning Repository.

Special thanks to the open-source ML community.

🔥 Let's detect cancer early and save lives! 🔥
