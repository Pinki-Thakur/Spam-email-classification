import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()



def clean_text(text):
    # convert to lowercase
    text = text.lower()
    # tokenize
    text = nltk.word_tokenize(text)

    # remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # remove stopwords
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # remove extra spaces
    # text = re.sub(r'\s+', ' ', text)
    # stemming
    ps = PorterStemmer()
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf= pickle.load(open('vectorizer.pkl', 'rb'))
model= pickle.load(open('spam_classification_model.pkl', 'rb'))

st.title("Email Spam Classifier")

input_email= st.text_area(" Enter the Email")

if st.button('Predict'):

    # preprocess
    transformed_email= clean_text(input_email)
    #vectorize
    vector_input= tfidf.transform([transformed_email])
    #predict
    result= model.predict(vector_input)
    #Display
    if result== 1:
        st.header("Spam")
    else:
         st.header("Not Spam")









