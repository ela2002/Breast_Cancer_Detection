import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
csv_file_path = "wdbc.data"  
column_names = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]
data_frame = pd.read_csv(csv_file_path, header=None, names=column_names)

# Inspecting the dataset
logging.info("Dataset loaded successfully from CSV.")
logging.info(f"Dataset Shape: {data_frame.shape}")
logging.info("Dataset Info:")
logging.info(data_frame.info())
logging.info("Dataset Description:")
logging.info(data_frame.describe())

# Extracting labels (Diagnosis column) and dropping the ID column
labels = data_frame['diagnosis'].map({'M': 1, 'B': 0})  # Convert M to 1 (Malignant) and B to 0 (Benign)
data_frame = data_frame.drop(columns=['id', 'diagnosis'])

# Visualizing Class Distribution
sns.countplot(x=labels)
plt.title("Class Distribution (Malignant vs Benign)")
plt.show()

# Checking for Missing Values
logging.info("Checking for missing values...")
logging.info(data_frame.isnull().sum())

# Checking for duplicate rows
logging.info("Checking for duplicate rows...")
logging.info(f"Duplicate Rows: {data_frame.duplicated().sum()}")

# Correlation Heatmap to visualize relationships between features
plt.figure(figsize=(12, 10))
sns.heatmap(data_frame.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(data_frame, labels, test_size=0.2, random_state=42, stratify=labels)
logging.info("Data split into training and testing sets.")

# Handling Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
logging.info("Applied SMOTE to balance the dataset.")

# Standardizing Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logging.info("Feature scaling completed.")

# Models to Evaluate
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "KNN": KNeighborsClassifier()
}

# Model Training and Evaluation
results = {}
plt.figure(figsize=(10, 5))

for name, model in models.items():
    logging.info(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    logging.info(f"Accuracy for {name}: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix for {name}")
    plt.show()

    # Classification Report
    logging.info(f"Classification Report for {name}:\n%s", classification_report(y_test, y_pred))

    # ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Comparing Model Performance
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

# Feature Importance (Random Forest)
rf = models["Random Forest"]
feature_importances = pd.Series(rf.feature_importances_, index=data_frame.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.show()

logging.info("Model evaluation completed and results visualized.")
