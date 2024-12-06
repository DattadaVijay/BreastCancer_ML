import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("data.csv")

X = data.drop(columns=['diagnosis']) 
y = data['diagnosis']
X = X.drop(columns=['Unnamed: 32'])

# Handle missing values in X using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

####################################################################################################
# 1. Correlation Matrix (Filter Method)
####################################################################################################

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# Identifying pairs of highly correlated features (correlation > 0.9)
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

print(f"Highly correlated features to drop: {highly_correlated_features}")

####################################################################################################
# 2. Recursive Feature Elimination (RFE) - Wrapper Method
####################################################################################################

# Initializing a RandomForest model
rf = RandomForestClassifier()
rfe = RFE(estimator=rf, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y_encoded)
selected_features_rfe = X.columns[rfe.support_]
print("Selected features from RFE:", selected_features_rfe)

####################################################################################################
# 3. Lasso Regression (Embedded Method)
####################################################################################################

# Initializing LassoCV model
lasso = LassoCV(cv=5)
lasso.fit(X, y_encoded)
lasso_coefficients = pd.Series(lasso.coef_, index=X.columns)

# Display the features with non-zero coefficients
selected_lasso_features = lasso_coefficients[lasso_coefficients != 0]
print("Selected features from Lasso regression:", selected_lasso_features)

####################################################################################################
# 4. Feature Importance from Random Forest (Embedded Method)
####################################################################################################
rf.fit(X, y_encoded)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
sorted_feature_importances = feature_importances.sort_values(ascending=False)
print("Top 10 features by importance from RandomForest:")
print(sorted_feature_importances.head(10))

####################################################################################################
# Summary of Feature Selection Methods
####################################################################################################

# Correlation-based feature removal (features to drop from the correlation matrix)
print("\nFeatures to drop based on correlation matrix:")
print(highly_correlated_features)

# RFE-selected features
print("\nSelected features from RFE method:")
print(selected_features_rfe)

# Lasso-selected features
print("\nSelected features from Lasso regression method:")
print(selected_lasso_features)

# Top features from RandomForest importance
print("\nTop features from RandomForest importance:")
print(sorted_feature_importances.head(10))
