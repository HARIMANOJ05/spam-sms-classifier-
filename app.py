import streamlit as st
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps=PorterStemmer()


def transform_text(text):
    text= text.lower()
    text= nltk.word_tokenize(text)
    
    y=[]
     
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english')and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
        
    
    return " ".join(y)



tfvec=pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("SPAM MESSAGE CLASSIFIER APPLICATION")

input_sms=st.text_area('Enter the message')


if st.button('predict'):

#preprocess

        transformed_sms= transform_text(input_sms)

#vectorizer

        vector_input= tfvec.transform([transformed_sms])
        dense_vector_input= vector_input.toarray()
#predict 

        result=model.predict(dense_vector_input)[0]

#display

        if result==1:
                st.header("Spam")
        else:
            st.header("Not Spam")