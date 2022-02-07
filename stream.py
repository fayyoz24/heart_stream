import streamlit as st
import data_handler as dh
import pandas as pd
import numpy as np
from PIL import Image

from train import Models
import moduls as mls


st.markdown("<h1 style='text-align: center; color: coral;'>You can not live without HEART!...</h1>", unsafe_allow_html=True)
img = Image.open("heart.jpg")
st.image(img, width=700)

df = pd.read_csv('heart.csv', index_col=0)

data_ = st.sidebar.radio(
    label="Do you want to see the whole data?",
    options=("No",'Yes'))

if data_ == 'Yes':
    st.markdown("<h1 style='text-align: center; color: red;'>The given data!</h1>", unsafe_allow_html=True)
    st.write(df)

st.sidebar.subheader("Alghorithms")
top_book_ = st.sidebar.selectbox(
    label ="Select your Algorithm",
    options = ['SVC','LogisticRegression','RandomForestClassifier','KNeighborsClassifier',
    'GradientBoostingClassifier','XGBClassifier','DecisionTreeClassifier'])

models = [Models.svc(), Models.lr(), Models.rfc(), Models.knc(), Models.gbc(), Models.xgbc(), Models.dtc()]

a = ['SVC','LogisticRegression','RandomForestClassifier','KNeighborsClassifier',
    'GradientBoostingClassifier','XGBClassifier','DecisionTreeClassifier']

if top_book_ :
    st.text('''
    
    


    ''')
    st.write(models[a.index(top_book_)])

try:
    user_input = st.sidebar.radio(
    label="Do you want to know about your healthy?",
    options=("No",'Yes'))
    if user_input == 'Yes':
        
        st.title('Disease Predictor')
        text = """
            <div style = 'background-color:tomato;padding:10px'>
            <h4 style= 'color:white;text-align:center;'>Streamlit Heart Disease Predictor ML App </h4>
            </div>
        """
        st.markdown(text,unsafe_allow_html=True)
        st.title('')
        st.markdown('<h5> Please Answer following questions!</h5>', unsafe_allow_html=True)
        st.title('')

        age = st.text_input("How old are you? \n")

        sex_ = st.radio(label='Choose your gender', options=('Male', 'Female'))
        if sex_ == 'Male':
            sex = 1
        else:
            sex = 0
        
        cp = st.radio(label='Choose your CP', options=(0,1,2,3))

        trtbps = st.text_input("Enter yout TRTBPS\n")

        chol = st.text_input("Enter your CHOL \n")

        fbs = st.radio(label='Choose your FBS', options=(0,1))

        restecg = st.radio(label='Choose your RESTESG', options=(0,1,2))

        thalc = st.text_input("Enter your THALAShC \n")

        exng = st.radio(label='Choose your EXNG', options=(0,1))

        oldpeak = st.text_input("Enter your OLDPEAK \n")

        slp = st.radio(label='Choose your SLP', options=(0,1,2))

        caa = st.radio(label='Choose your CAA', options=(0,1,2,3, 4))

        thall = st.radio(label='Choose your THALL', options=(0,1,2,3))


        x = pd.DataFrame({"age":age, "sex":sex, 'cp':cp, 'trtbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalachh':thalc,
            'exng':exng, 'oldpeak':oldpeak, 'slp':slp, 'caa':caa, 'thall':thall}, index=[0])

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./heart.csv')
        x_scaled  = scaler.transform(x)
        predictions = np.array([model.predict(x_scaled) for model in mls.models()])

        predict = st.button("predict")
        if predict:   
            if predictions.mean() < 0.5:
                st.markdown("<h3 style='text-align: center; color: blue;'>You are Healthy!!!</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: blue;'>Please go to the doctor...\n I got POSITIVE result</h3>", unsafe_allow_html=True)

except ValueError:
    st.error('Please enter valid data type!')

about = st.sidebar.button("contributors")
if about:
    st.title("Fayyozjon Usmonov")
    st.markdown("<h1 style='text-align: center; color: blue;'>Thanks for your attention!</h1>", unsafe_allow_html=True)