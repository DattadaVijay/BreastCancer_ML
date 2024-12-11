import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

################################################################### Handling the outliers #############################################################################################
data_with_outliers = pd.read_csv("data.csv")
print(data_with_outliers.dtypes)
features = data_with_outliers.select_dtypes(include=float)

mean = features.mean()
std_dev = features.std()

outlier_mask = (data_with_outliers.select_dtypes(include=["Float64"]) < (mean - 3 * std_dev)) | (data_with_outliers.select_dtypes(include=["Float64"]) > (mean + 3 * std_dev))
outliers = data_with_outliers[outlier_mask.any(axis=1)]
data_without_outliers = data_with_outliers[~outlier_mask.any(axis=1)]

print(data_with_outliers.head(5))
print(data_without_outliers.head(5))

# We are gonna Explore the modelling for both of these data.

#################################################################### Exploratory Data Analysis ######################################################################################
# Map diagnosis column and drop unnecessary columns
data_with_outliers['diagnosis'] = data_with_outliers['diagnosis'].map({'M': 1, 'B': 0})
data_with_outliers = data_with_outliers.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Combine features and target into a single DataFrame
X_with_target = data_with_outliers.copy()

# After exploration, spliting the data into X and y
X = X_with_target.drop(columns=['diagnosis'])
y = X_with_target['diagnosis']

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# 3. Pairplots to see the feature nature
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Splitting the feature columns into subsets of 5 features each (for better visibility)
feature_subsets = [feature_columns[i:i + 5] for i in range(0, len(feature_columns), 5)]

# Loop through feature_subsets to create and display separate pairplots
for i, subset in enumerate(feature_subsets):
    # Generate the pairplot for the selected subset, using X_with_target for diagnosis
    sns.pairplot(X_with_target[subset + ['diagnosis']], hue='diagnosis', height=2.5)
    
    # Adjust the title for each pairplot
    plt.suptitle(f'Pairplot of Features: {", ".join(subset)}', fontsize=16)
    
    # Adjust the layout for better display
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top space to avoid title overlap
    
    # Display the pairplot
    plt.show()  # Display the plot interactively
