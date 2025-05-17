from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
if os.path.exists("emotion_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("emotion_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    raise Exception("Model or vectorizer file is missing or corrupted.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return render_template('index.html', prediction=prediction[0], text=text)

if __name__ == '__main__':
    app.run(debug=True)
