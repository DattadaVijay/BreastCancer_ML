import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv("data.csv")

# Preprocess the data
X = data.drop(columns=['diagnosis', 'Unnamed: 32'], errors='ignore')
y = data['diagnosis']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling Methods (Standard, Min-Max)
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'No Scaling': None
}

# Models (including ensemble models and XGBoost)
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Voting Classifier': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('lr', LogisticRegression(max_iter=1000)),
            ('svc', SVC(probability=True))
        ], voting='soft'),
    'Bagging Classifier': BaggingClassifier(n_estimators=50),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Store results for comparison and ROC data
results_experiment = {}
roc_data = []

# Model Evaluation Function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if hasattr(model, 'predict_proba') else None
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc, conf_matrix, y_pred_prob

# Run the experiment for each combination of scaling, PCA, and model
for model_name, model in models.items():
    for scaling_name, scaler in scalers.items():
        for apply_pca in [True, False]:
            pca_suffix = "with PCA" if apply_pca else "without PCA"
            print(f"\nEvaluating {model_name} with {scaling_name} scaling {pca_suffix}")

            # Apply scaling (if applicable)
            if scaler:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Apply PCA (if applicable)
            if apply_pca:
                pca = PCA(n_components=min(X_train_scaled.shape[1], 10))
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)

            # Evaluate the model
            acc, prec, rec, f1, auc, cm, y_pred_prob = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Store ROC data if AUC is computable
            if auc is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                roc_data.append({
                    'Model': model_name,
                    'Scaling': scaling_name,
                    'PCA': apply_pca,
                    'FPR': fpr,
                    'TPR': tpr,
                    'AUC': auc
                })

            # Store results in a DataFrame
            temp_df = pd.DataFrame([{"Model": model_name, "Scaling": scaling_name, "PCA": apply_pca,
                                     "Accuracy": acc, "Precision": prec, "Recall": rec, 
                                     "F1 Score": f1, "ROC AUC": auc}], 
                                   columns=["Model", "Scaling", "PCA", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
            
            if model_name not in results_experiment:
                results_experiment[model_name] = []
            results_experiment[model_name].append(temp_df)

# Concatenate all results into a single DataFrame
result_df = pd.concat([pd.concat(results) for results in results_experiment.values()], ignore_index=True)

# Display the result DataFrame
print(result_df)

# Print the best model based on ROC AUC
best_model = result_df.loc[result_df['ROC AUC'].idxmax()]
print("\nBest Model Based on ROC AUC:")
print(best_model)

# Plot ROC Curves for all models
plt.figure(figsize=(12, 8))
for data in roc_data:
    label = f"{data['Model']} ({data['Scaling']}, {'PCA' if data['PCA'] else 'No PCA'}: AUC={data['AUC']:.4f})"
    plt.plot(data['FPR'], data['TPR'], label=label)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')

# Place the legend outside the plot
plt.legend(loc="lower right", fontsize='small', title="Models", ncol=2)


plt.tight_layout()
plt.show()
