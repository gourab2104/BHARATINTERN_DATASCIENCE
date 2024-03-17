import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Explore the dataset
print(titanic_data.head())
print(titanic_data.info())

# Data preprocessing
# Drop unnecessary columns
titanic_data.drop(['deck', 'embark_town', 'alive'], axis=1, inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
titanic_data['age'] = imputer.fit_transform(titanic_data['age'].values.reshape(-1, 1))

# Feature engineering
titanic_data['family_size'] = titanic_data['sibsp'] + titanic_data['parch']

# Prepare features and target variable
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical features
numeric_features = ['age', 'fare', 'parch', 'sibsp', 'family_size']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['sex', 'class', 'embarked', 'who', 'adult_male']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualizations
# Bar plot for survival counts
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', data=titanic_data)
plt.title('Survival Counts')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Pie chart for class distribution
plt.figure(figsize=(8, 6))
titanic_data['class'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Class Distribution')
plt.ylabel('')
plt.show()

# Histogram of age distribution
plt.figure(figsize=(8, 6))
sns.histplot(titanic_data['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot of fare by survival
plt.figure(figsize=(8, 6))
sns.boxplot(x='survived', y='fare', data=titanic_data)
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()
