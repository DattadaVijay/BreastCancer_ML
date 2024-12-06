import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("data.csv")

# Preprocess the data (assuming 'diagnosis' is the target column)
X = data.drop(columns=['diagnosis', 'Unnamed: 32'])
y = data['diagnosis']

# Convert target labels to binary (M = 1, B = 0)
y = y.map({'M': 1, 'B': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

####################################################################################################
# Feature Selection using Random Forest
####################################################################################################

# Initialize RandomForest for feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf.feature_importances_

# Select features with non-zero importance
selected_features = X.columns[importances > 0]
X_train_selected = X_train_scaled[:, importances > 0]
X_test_selected = X_test_scaled[:, importances > 0]

####################################################################################################
# Model Evaluation Function
####################################################################################################

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Update pos_label to 'M' (malignant)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)  # Set pos_label to 1 (malignant)
    recall = recall_score(y_test, y_pred, pos_label=1)  # Set pos_label to 1 (malignant)
    f1 = f1_score(y_test, y_pred, pos_label=1)  # Set pos_label to 1 (malignant)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, roc_auc, conf_matrix

####################################################################################################
# Train and Evaluate Models
####################################################################################################

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_acc, rf_prec, rf_recall, rf_f1, rf_roc_auc, rf_conf_matrix = evaluate_model(rf_model, X_train_selected, X_test_selected, y_train, y_test)
print(f"Random Forest - Accuracy: {rf_acc:.4f}, Precision: {rf_prec:.4f}, Recall: {rf_recall:.4f}, F1: {rf_f1:.4f}, ROC AUC: {rf_roc_auc:.4f}")
print(f"Confusion Matrix:\n{rf_conf_matrix}\n")

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_acc, xgb_prec, xgb_recall, xgb_f1, xgb_roc_auc, xgb_conf_matrix = evaluate_model(xgb_model, X_train_selected, X_test_selected, y_train, y_test)
print(f"XGBoost - Accuracy: {xgb_acc:.4f}, Precision: {xgb_prec:.4f}, Recall: {xgb_recall:.4f}, F1: {xgb_f1:.4f}, ROC AUC: {xgb_roc_auc:.4f}")
print(f"Confusion Matrix:\n{xgb_conf_matrix}\n")

# SVM
svm_model = SVC(probability=True, random_state=42)
svm_acc, svm_prec, svm_recall, svm_f1, svm_roc_auc, svm_conf_matrix = evaluate_model(svm_model, X_train_selected, X_test_selected, y_train, y_test)
print(f"SVM - Accuracy: {svm_acc:.4f}, Precision: {svm_prec:.4f}, Recall: {svm_recall:.4f}, F1: {svm_f1:.4f}, ROC AUC: {svm_roc_auc:.4f}")
print(f"Confusion Matrix:\n{svm_conf_matrix}\n")

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_acc, logreg_prec, logreg_recall, logreg_f1, logreg_roc_auc, logreg_conf_matrix = evaluate_model(logreg_model, X_train_selected, X_test_selected, y_train, y_test)
print(f"Logistic Regression - Accuracy: {logreg_acc:.4f}, Precision: {logreg_prec:.4f}, Recall: {logreg_recall:.4f}, F1: {logreg_f1:.4f}, ROC AUC: {logreg_roc_auc:.4f}")
print(f"Confusion Matrix:\n{logreg_conf_matrix}\n")

# Entropy-based model (Decision Tree with Entropy criterion)
from sklearn.tree import DecisionTreeClassifier
entropy_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
entropy_acc, entropy_prec, entropy_recall, entropy_f1, entropy_roc_auc, entropy_conf_matrix = evaluate_model(entropy_model, X_train_selected, X_test_selected, y_train, y_test)
print(f"Decision Tree (Entropy) - Accuracy: {entropy_acc:.4f}, Precision: {entropy_prec:.4f}, Recall: {entropy_recall:.4f}, F1: {entropy_f1:.4f}, ROC AUC: {entropy_roc_auc:.4f}")
print(f"Confusion Matrix:\n{entropy_conf_matrix}\n")

# Likelihood-based model (Naive Bayes)
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_acc, nb_prec, nb_recall, nb_f1, nb_roc_auc, nb_conf_matrix = evaluate_model(nb_model, X_train_selected, X_test_selected, y_train, y_test)
print(f"Naive Bayes - Accuracy: {nb_acc:.4f}, Precision: {nb_prec:.4f}, Recall: {nb_recall:.4f}, F1: {nb_f1:.4f}, ROC AUC: {nb_roc_auc:.4f}")
print(f"Confusion Matrix:\n{nb_conf_matrix}\n")
