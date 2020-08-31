from flask import Flask, request, render_template
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from model import InputCommentForm

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputCommentForm(request.form)
    if request.method == 'POST' and form.validate():
        user_input = form.r.data
        model = load('sklearn_model.joblib')
        vectorizer = load('vectorizer.joblib')
        text = vectorizer.transform([user_input]) 
        prediction = model.predict(text)[0]
        if prediction == 0:
            s = 'not spam'
        else:
            s = 'spam'
    else:
        s = None

    return render_template("view.html", form=form, s=s)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


