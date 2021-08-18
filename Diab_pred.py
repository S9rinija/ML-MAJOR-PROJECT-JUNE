import numpy as np
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('desc.csv')
df1 = pd.read_csv('diabetes.csv')
pickle_in = open("Maj_proj_model_pickle", "rb")
classifier = pickle.load(pickle_in)
def Diabetes_prediction(Preg,Gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    prediction = classifier.predict([[Preg,Gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(prediction)
    return prediction
def main():
    st.write("""
    # Diabetes Prediction App
    """)
    st.header('User Input Features')
    html_temp = """
    <div style="background-color:#546beb;padding:10px">
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)    
    Preg =st.slider("Pregnancies",0,20,6)
    Gluc = st.slider("Glucose",0,200,140)
    BP = st.slider("BloodPressure",0,120,80)
    SkinThickness = st.slider("SkinThickness",0,99,32)
    Insulin = st.slider("Insulin",0,846,127)
    BMI = st.slider("BMI",0.0,70.0,35.0)
    DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFunction",0.078,2.42,0.62)
    Age = st.slider("Age",21,81,41)
    
    result = ""
    if st.button("Predict"):
        result = Diabetes_prediction(int(Preg),int(Gluc),int(BP),int(SkinThickness),int(Insulin),float(BMI), float(DiabetesPedigreeFunction),int(Age))
        if result[0]==0 :
            st.success("Good news! It looks like you don't have diabetes.")
        else :
            st.error("Sorry, it looks like you have diabetes.")

    sns.boxplot(x='Outcome',y='BMI',data=df1)
    st.pyplot()
    if st.button("About our data"):
        st.write(df.set_index('Unnamed: 0'))
#if __name__ == '__main__':
main()
