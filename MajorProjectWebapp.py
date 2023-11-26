!pip install streamlit
!pip install numpy 
!pip install pandas
!pip install sklearn
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
st.title("Review Prediction")
# Creating a dataframe
df=pd.read_table("https://raw.githubusercontent.com/gudhi987/MajorProject/main/Restaurant_Reviews.tsv")
#divide data into input and output
x=df['Review'].values
y=df['Liked'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)
x_train_vect.toarray() #gives a sparse representation of numbers after converting from text 
from sklearn.pipeline import make_pipeline 
model = make_pipeline(CountVectorizer(),SVC())
model.fit(x_train,y_train) 
input=[]
input_string=st.text_input(label='Enter your review')
input.append(input_string)
y_pred=model.predict(input)
op = ['Negative','Positive']
st.write('The review provided is:',op[y_pred[0]])
