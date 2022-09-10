import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,LogisticRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,Normalizer
cancer=pd.read_csv("cancer.csv")
radius=st.sidebar.number_input("Mean Radius")
texture=st.sidebar.number_input("Mean Texture")
perimeter=st.sidebar.number_input("Mean Perimeter")
area=st.sidebar.number_input("Mean Area")
smoot=st.sidebar.number_input("Mean Smoothness")
solversec=st.sidebar.selectbox("Choose Solver",("newton-cg","liblinear","saga","sag","lbfgs"))
hesapla=st.sidebar.button("Hesapla")
if hesapla:
    y=cancer[["diagnosis"]]
    x=cancer.drop("diagnosis",axis=1)
    log=LogisticRegression(solver=solversec)
    model=log.fit(x,y)
    skor=model.score(x,y)
    tahmin=model.predict([[radius,texture,perimeter,area,smoot]])[0]
    if tahmin==0:
         st.write("test sonucu negatif")
    elif tahmin==1:
        st.write("test sonucu pozitif")
    st.write("test tahmin başarı oranı",skor)