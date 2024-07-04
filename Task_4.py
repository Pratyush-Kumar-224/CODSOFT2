import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Explore dataset if needed: data.head(), data.info(), etc.

# Preprocess the data
# Assuming the dataset has columns 'label' for spam/ham and 'text' for SMS content
X = data['text']
y = data['label']

# Convert labels to binary values: spam as 1, ham as 0
y = y.apply(lambda x: 1 if x == 'spam' else 0)

# Extract features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Support Vector Machine")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
