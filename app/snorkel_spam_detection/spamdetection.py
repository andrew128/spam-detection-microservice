from snorkel.labeling import PandasLFApplier
from snorkel_spam_detection import utils, labelingfunctions
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from sklearn.feature_extraction.text import CountVectorizer
from snorkel.utils import probs_to_preds
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

class SpamDetection:
    '''
    Train and save an logistic regression model trained using Snorkel
    '''
    def __init__(self):
        self.lfs = [
            labelingfunctions.keyword_my,
            labelingfunctions.keyword_subscribe,
            labelingfunctions.keyword_link,
            labelingfunctions.keyword_please,
            labelingfunctions.keyword_song,
            labelingfunctions.regex_check_out,
            labelingfunctions.short_comment,
            labelingfunctions.has_person_nlp,
            labelingfunctions.textblob_polarity,
            labelingfunctions.textblob_subjectivity,
        ]

        self.df_train, self.df_test = utils.load_spam_dataset()

        self.train()

    def addLfs(self, word):
        '''
        Adds new lf function but does not retrain

        Input: string word
        '''
        self.lfs.append(labelingfunctions.make_keyword_lf(keywords=[word]))

    def train(self):
        '''
        Train the logistic regression discriminative model
        '''
        # We pull out the label vectors for ease of use later
        Y_test = self.df_test.label.values

        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df=self.df_train)

        # Use Label Model to combined input data
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        # Make predictions
        probs_train = label_model.predict_proba(L=L_train)

        # Filter abstained inputs
        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=self.df_train, y=probs_train, L=L_train
        )

        # Represent each data point as a one-hot vector
        vectorizer = CountVectorizer(ngram_range=(1, 5))
        X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
        X_test = vectorizer.transform(self.df_test.text.tolist())

        # Turn probs into preds
        preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

        # Train logistic regression model
        sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
        sklearn_model.fit(X=X_train, y=preds_train_filtered)

        print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")
        dump(sklearn_model, 'sklearn_model.joblib')
        dump(vectorizer, 'vectorizer.joblib')

    def predict(self, user_input):
        spam_model = load('./sklearn_model.joblib')
        vectorizer = load('./vectorizer.joblib')
        text = vectorizer.transform([user_input]) 
        prediction = spam_model.predict(text)[0]
        if prediction == 0:
            return 'not spam'
        else:
            return 'spam'

def main(): 
    spamDetection = SpamDetection()
    dump(spamDetection, './spamDetection.joblib')
    sd = load('./spamDetection.joblib')
    print(sd.predict('check it out'))

if __name__=="__main__": 
    main() 
