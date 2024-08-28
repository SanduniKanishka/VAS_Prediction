from msilib.schema import File
import sklearn
import streamlit as st
import pickle
import numpy as np
from PIL import Image

primaryColor = "#E694FF"
backgroundColor = "#00172B"
secondaryBackgroundColor = "#0083B8"
textColor = "#C6CDD4"
font = "sans-serif"

def load_model(pickleName):
    with open(pickleName, 'rb') as file:
        data = pickle.load(file)
    return data


def show_predictPage():
    st.title("VAS service Prediction")
    #st.write("############ggggg")

    ages = ("51-60",
           "41-50",
           "31-40",
           "21-30",
           "Under 20",)

    connection_age = ("more than 5 years",
                       "3-5 years",
                       "1-3 years",
                       "less than 1",)

    devices = ("SMART", 
               "BASIC", 
               "FEATURE",)
    
    languages = ("English",
                 "Sinhala",
                 "Tamil",)

    gender = ("Female",
              "Male",)
    algorithms1=("Random Forest",
                "Bagged CART",
                "Stacking",
                "Logistic Regression",
                "Boosting",)

    c5, c6= st.columns(2)
    with c5:
         age = st.selectbox("Age Category", ages)
    with c6:
        conn = st.selectbox("Connection", connection_age)

    c7, c8= st.columns(2)
    with c7:
         device = st.selectbox("Device", devices)
    with c8:
        language = st.selectbox("Language", languages)
    
    c9, c10= st.columns(2)
    with c9:
         gender1 = st.selectbox("Gender", gender)      
    with c10:
        algorithms = st.selectbox("Algorithm",algorithms1)
    
    
    c1, c2= st.columns(2)
    with c1:
        voiceUsage = st.text_input("Voice Usage")
    with c2:
        dataUsage = st.text_input("Data Usage")

    c3, c4= st.columns(2)
    with c3:
        revenue = st.text_input("Revenue")
    with c4:
        vasRevenue = st.text_input("VAS Revenue")


    if(gender1 == "Male"):
        genderCustomer = 1
    else:
        genderCustomer = 0
    
    if(algorithms == "Random Forest" ):
        pickleSelected = "randomForest1.pkl"
    if(algorithms == "Bagged CART" ):
        pickleSelected = "baggedcart.pkl"
    if(algorithms == "Stacking" ):
        pickleSelected = "stack.pkl"
    if(algorithms == "Logistic Regression" ):
        pickleSelected = "logisticRegression.pkl"
    if(algorithms == "Boosting" ):
        pickleSelected = "xgb_clf.pkl"

    data = load_model(pickleSelected)

    regressor = data["model"]
    le_age = data["le_age"]
    le_device = data["le_device"]
    le_language = data["le_language"]
    le_conAge = data["le_conAge"]
    le_package=data["le_package"]


    ok = st.button("Predict Service")
    if ok:
        X = np.array([[genderCustomer, age, conn,language, voiceUsage,dataUsage,device,revenue,vasRevenue]])
        X[:, 1] = le_age.transform(X[:,1])
        X[:, 2] = le_conAge.transform(X[:,2])
        X[:, 3] = le_language.transform(X[:,3])
        X[:, 6] = le_device.transform(X[:,6])
        print(X)
        X = X.astype(float)
        print(X)
        pred = regressor.predict(X)
        #st.write("VAS service11: {}".format(pred[0]))
        prediction = le_package.inverse_transform(pred)
        st.write("VAS service: {}".format(prediction[0]))
        #return render_template('after.html', data=pred)

    img = Image.open("Accuracy.jpg")
    st.image(img)