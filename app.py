import streamlit as st
import requests as r
from os.path import dirname, join, realpath
import joblib
from langdetect import detect

# text preprocessing modules
from string import punctuation

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression

# load stop words
stop_words = stopwords.words("english")


# add banner image
st.header("Gender-Based Violence Tweet Classification App")
st.image("images/Stop-Gender-based-Violence.png")
st.subheader(
    """
A  Data science app to classify tweets about GBV without using keywords.
"""
)

# form to collect news content
my_form = st.form(key="tweets_form")
tweet = my_form.text_input("Input your tweet here")
submit = my_form.form_submit_button(label="make prediction")


# load the model and count_vectorizer

with open(join(dirname(realpath(__file__)), "models/tweets_model.pkl"), "rb") as f:
    model = joblib.load(f)

with open(join(dirname(realpath(__file__)), "preprocessors/vectorizer.pkl"), "rb") as f:
    vectorizer = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "preprocessors/labelEncoder.pkl"), "rb"
) as f:
    labelEncoder = joblib.load(f)


# function to clean the text

# clean the dataset


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    text = text.lower()

    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text


if submit:

    if detect(tweet) == "en":

        cleaned_tweet = text_cleaning(tweet)

        # transform the input
        transformed_tweet = vectorizer.transform([tweet])

        # perform prediction
        prediction = model.predict(transformed_tweet)

        output = int(prediction[0])
        probas = model.predict_proba(transformed_tweet)
        probability = "{:.2f}".format(float(probas[:, output]))

        class_predicted = labelEncoder.inverse_transform([prediction[0]])

        # Display results of the NLP task
        st.header("Results")

        st.write("The tweet has {} content".format(class_predicted[0]))

    else:
        st.write(
            " ⚠️ The tweet is not in English language.Please make sure the input is in English language"
        )

url = "https://twitter.com/Davis_McDavid"
st.write("Developed with ❤️ by [Davis David](%s)" % url)
