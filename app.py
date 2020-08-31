from flask import Flask, request, render_template
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('input-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    model = load('sklearn_model.joblib')
    vectorizer = load('vectorizer.joblib')
    text = vectorizer.transform([text]) 
    # print(model.predict(text))
    return 'Prediction is: ' + str(model.predict(text)[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


