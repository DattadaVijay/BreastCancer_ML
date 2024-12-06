import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("data.csv")

# Understanding the data
'''
R - Reliability
O - Originality (Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic, 10.24432/C5DW2B)
C - Comprehensiveness (The data has 63% Benign and 37% Malignant - A good data) 
C - Citations (It has been academically cited in 37 research Publications)
'''

###################################################################### Test for realiability#######################################################################################

# 1. Check for missing values 

missing_values = data.isna().sum()
duplicate_values = data.duplicated().sum()
reliability = {"missing_values": missing_values, "duplicate_values": duplicate_values}
print(reliability) # 0 Missing values and 0 duplicates

# 2. Further checking the skewness of the data. 

features = list(data.columns)
print(features)


for feature in features[2:32]:
    mean = data[feature].mean()
    std_dev = data[feature].std()
    plt.figure(figsize=(8, 6))
    plt.hist(data[feature], bins=30, edgecolor='black', alpha=0.7)

    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')

    plt.axvline(mean + std_dev, color='blue', linestyle='dashed', linewidth=2, label=f'+1 SD: {mean + std_dev:.2f}')
    plt.axvline(mean - std_dev, color='blue', linestyle='dashed', linewidth=2, label=f'-1 SD: {mean - std_dev:.2f}')

    plt.axvline(mean + 2*std_dev, color='green', linestyle='dashed', linewidth=2, label=f'+2 SD: {mean + 2*std_dev:.2f}')
    plt.axvline(mean - 2*std_dev, color='green', linestyle='dashed', linewidth=2, label=f'-2 SD: {mean - 2*std_dev:.2f}')

    plt.axvline(mean + 3*std_dev, color='purple', linestyle='dashed', linewidth=2, label=f'+3 SD: {mean + 3*std_dev:.2f}')
    plt.axvline(mean - 3*std_dev, color='purple', linestyle='dashed', linewidth=2, label=f'-3 SD: {mean - 3*std_dev:.2f}')

    plt.title(f'Histogram of {feature} with Mean and Standard Deviation Lines')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{feature}.png", bbox_inches='tight')
    plt.close()

# All the plots have been analysed based on the means and the standard deviations the data distribution is very reliable and not skewed. 
# There were few outliers identified but that in modelling can help identify very intersting patterns.
# There fore this data is reliable

# The data has passed our ROCCC criteria.