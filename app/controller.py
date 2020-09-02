from flask import Flask, request, render_template
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from snorkel_spam_detection import spamdetection
from model import CommentForm, NewLFForm
import os
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.secret_key = os.urandom(24)

spamDetection = spamdetection.SpamDetection()

@app.route('/', methods=['GET', 'POST'])
def index():
    comment_form = CommentForm()
    new_lf_form = NewLFForm()

    s = None
    if comment_form.validate_on_submit():
        user_input = comment_form.comment.data
        s = spamDetection.predict(user_input)

    if new_lf_form.validate_on_submit():
        user_input = new_lf_form.word.data
        spamDetection.addLfs(user_input)
        spamDetection.train()

    return render_template("view.html", commentform = comment_form, newlfform=new_lf_form, s = s)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


