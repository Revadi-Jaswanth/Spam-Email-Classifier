import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'text': [
        'Win a free iPhone now', 
        'Limited offer just for you', 
        'Hey, how are you?', 
        'Call me when you are free', 
        'Congratulations! You won a prize',
        'Important update about your account',
        'Let’s catch up tomorrow at lunch',
        'Claim your free vacation today'
    ],
    'label': [1, 1, 0, 0, 1, 0, 0, 1]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Try new email prediction
sample = ["Congratulations! You've won a new car!"]
sample_vec = vectorizer.transform(sample)
print("Spam Prediction:", model.predict(sample_vec)[0])  # 1 = spam
