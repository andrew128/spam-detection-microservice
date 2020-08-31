from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('input-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    model = load('filename.joblib') 
    return model.predict([text])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


