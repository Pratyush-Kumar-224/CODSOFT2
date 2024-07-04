import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load datasets
train_data = pd.read_csv('train_data.txt', sep='\t', header=None, names=['genre', 'plot'])
test_data = pd.read_csv('test_data.txt', sep='\t', header=None, names=['genre', 'plot'])

# Preprocess the data: Convert genres to categorical labels
train_data['genre'] = train_data['genre'].astype('category')
train_data['genre_cat'] = train_data['genre'].cat.codes
test_data['genre'] = test_data['genre'].astype('category')
test_data['genre_cat'] = test_data['genre'].cat.codes

# Extract TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tfidf.fit_transform(train_data['plot'])
y_train = train_data['genre_cat']
X_test = tfidf.transform(test_data['plot'])
y_test = test_data['genre_cat']

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Naive Bayes")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=train_data['genre'].cat.categories)}")
