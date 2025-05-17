import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Kaggle emotion dataset (train and test)
train_df = pd.read_csv("data/train.txt", names=["text", "emotion"], sep=";")
test_df = pd.read_csv("data/test.txt", names=["text", "emotion"], sep=";")

# Vectorization
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

# Train model
model = MultinomialNB()
model.fit(X_train, train_df["emotion"])

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(test_df["emotion"], predictions)
print(f"✅ Model Accuracy on Test Data: {accuracy:.2f}")

# Save model & vectorizer
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")
