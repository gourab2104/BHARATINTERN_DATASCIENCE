import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from zipfile import ZipFile
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt

# Download the ZIP file and extract the CSV file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

with urllib.request.urlopen(url) as response:
    with ZipFile(BytesIO(response.read())) as zip_file:
        # Assuming the CSV file is named 'SMSSpamCollection'
        with zip_file.open('SMSSpamCollection') as csv_file:
            sms_data = pd.read_csv(csv_file, sep='\t', names=['label', 'message'])

# Explore the dataset
print(sms_data.head())

# Preprocess the data
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})  # Convert labels to numeric values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_data['message'], sms_data['label'], test_size=0.2, random_state=42)

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Display a bar chart of the label distribution
plt.figure(figsize=(8, 6))
sms_data['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Spam and Non-Spam Messages')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Print classification results
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
