from utils import load_spam_dataset
from snorkel.labeling import PandasLFApplier
from snorkel_spam_detection import keyword_my, keyword_subscribe, \
    keyword_link, keyword_please, keyword_song, regex_check_out, short_comment, \
    has_person_nlp, textblob_polarity, textblob_subjectivity
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from sklearn.feature_extraction.text import CountVectorizer
from snorkel.utils import probs_to_preds
from sklearn.linear_model import LogisticRegression
from joblib import dump

'''
Train and save an logistic regression model trained using Snorkel
'''

# Combine all labeling functions going to use
lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]

df_train, df_test = load_spam_dataset()

# We pull out the label vectors for ease of use later
Y_test = df_test.label.values

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

# Use Label Model to combined input data

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

# Make predictions
probs_train = label_model.predict_proba(L=L_train)

# Filter abstained inputs

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)

# Represent each data point as a one-hot vector

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())

# Turn probs into preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

# Train logistic regression model

sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)

print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")
dump(sklearn_model, 'sklearn_model.joblib')
dump(vectorizer, 'vectorizer.joblib')