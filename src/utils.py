import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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


