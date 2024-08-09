import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

dataset = pd.read_csv('train.csv')
dataset = dataset.drop('id', axis=1)

X = dataset.iloc[:, 1]
y = dataset['label']

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

X = X.apply(preprocess_text)

tweets = list(X)

X_train, X_test, y_train, y_test = train_test_split(tweets, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('model', MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

pickle.dump(pipeline, open('sentiment_model.pkl', 'wb'))










