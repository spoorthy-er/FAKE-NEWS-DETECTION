from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('fmd_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('fmd_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the news headline input from the form
        headline = request.form['headline']

        # Transform the headline using the loaded TF-IDF vectorizer
        transformed_headline = vectorizer.transform([headline])

        # Predict using the loaded model
        prediction = model.predict(transformed_headline)

        # Convert the prediction to a readable label
        result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake news"

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
