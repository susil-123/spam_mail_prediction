import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from PIL import Image
import streamlit as st
st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Spam mail Detection using Naive Bayes</p>', unsafe_allow_html=True)
img=Image.open('mail.jpg')
st.image(img,use_column_width=True)
d=pd.read_csv('spam.csv',encoding = "ISO-8859-1")
d.rename(columns={'v1':'result','v2':'text'},inplace=True)
d.drop('Unnamed: 3',axis=1,inplace=True)
d.drop('Unnamed: 4',axis=1,inplace=True)
d.drop('Unnamed: 2',axis=1,inplace=True)
res={"ham":0,"spam":1}
d.result=[res[i] for i in d.result]
d.drop_duplicates(inplace=True)
st.subheader('Data Set')
st.dataframe(d)
st.write("\n")
st.subheader("Data Info")
st.write(d.describe())
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=d['text']
y=d['result']
x=cv.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
import pickle
pickle.dump(model,open("spam.pkl","wb"))
pickle.dump(cv,open("vec.pkl","wb"))
clf=pickle.load(open("spam.pkl","rb"))
st.subheader("Input Area:")
msg=st.text_area("Enter the mail text here to find if it's a spam or not")
if msg !="":
    st.markdown(
        f"""User Input: {msg}"""
    )
st.write('\n')
pred=model.predict(x_test)
st.write('Accuracy: ',accuracy_score(y_test,pred)*100,"%")

data=[msg]
vect=cv.transform(data).toarray()
result=model.predict(vect)
st.subheader("Result: ")
st.write(result)
st.subheader("Spam ?")
def ans(result):
    if result==0:
        st.write("Nope")
    else:
        st.write("Yeah, its a spam")
st.write("\n")
ans(result)
