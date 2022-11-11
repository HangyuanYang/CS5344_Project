import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

text_clean_re = "^#\S+|\s#\S+|^@\S+|\s@\S+|www\.\S+|https?:\S+|http?:\S+"
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()


def stemming_and_lemmatization(text):
    tokens = []
    for token in text.split():
        if len(token.strip()) > 0:
            tokens.append(lemmatizer.lemmatize(stemmer.stem(token.strip())))
    return " ".join(tokens)


def remove_stopwords(text):
    text = text.strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(token)
    return " ".join(tokens)


def clean_re(text):
    text = re.sub(text_clean_re, ' ', text)
    return re.sub('[^A-Za-z]+', ' ', text)


def preprocess(df):
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: clean_re(x))
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
    df['text'] = df['text'].apply(lambda x: stemming_and_lemmatization(x))

    return df

#reference:https://www.kaggle.com/code/zayedhaque/twitter-sentiment-analysis#Preprocessing-the-Data
def model_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':15}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':15}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':20}, pad = 20)
