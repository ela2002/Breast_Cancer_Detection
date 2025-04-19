# Importations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Scikit-learn et autres librairies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Configuration du Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chargement du Dataset
csv_file_path = "wdbc.data"
column_names = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]
data = pd.read_csv(csv_file_path, header=None, names=column_names)

# Informations sur le dataset
logging.info("Dataset loaded successfully.")
logging.info(f"Shape: {data.shape}")
logging.info(data.info())
logging.info(data.describe())

# Prétraitement
labels = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(columns=['id', 'diagnosis'])

# Visualisation de la distribution des classes
sns.countplot(x=labels)
plt.title("Distribution des classes (Malin vs Bénin)")
plt.show()

# Vérification des valeurs manquantes et doublons
logging.info(f"Missing values:\n{data.isnull().sum()}")
logging.info(f"Duplicated rows: {data.duplicated().sum()}")

# Corrélation entre caractéristiques
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Heatmap des corrélations")
plt.show()

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)
logging.info("Données séparées en train et test.")

# Traitement des données déséquilibrées
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
logging.info("SMOTE appliqué.")

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logging.info("Données normalisées.")

# Définition des modèles
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Boosting": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Stacking": StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        final_estimator=LogisticRegression()
    ),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000)

}

# Entraînement et évaluation
results = {}
for name, model in models.items():
    logging.info(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Précision
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    logging.info(f"Accuracy for {name}: {acc * 100:.2f}%")

    # Matrice de confusion
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.show()

    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=report_df.index, y=report_df['f1-score'])
    plt.title(f"F1-Score par classe - {name}")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()

    # Courbe ROC
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    # Affichage de l'arbre pour Decision Tree
    if name == "Decision Tree":
        plt.figure(figsize=(20, 10))
        plot_tree(model, filled=True, feature_names=data.columns, class_names=["Benign", "Malignant"], max_depth=3)
        plt.title("Visualisation de l'Arbre de Décision (3 niveaux)")
        plt.show()

# Comparaison finale des performances
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Comparaison des performances des modèles")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()

# Importance des variables (Random Forest)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=data.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Importance des caractéristiques (Random Forest)")
plt.xlabel("Importance")
plt.show()

logging.info("Évaluation de tous les modèles terminée.")
