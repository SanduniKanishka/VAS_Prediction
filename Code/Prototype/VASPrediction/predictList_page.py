from cProfile import label
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import pip
from PIL import Image

#pip.main(["install", "openpyxl"])

def load_model(pickleName):
    with open(pickleName, 'rb') as file:
        data = pickle.load(file)
    return data



def show_predictListPage():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("Predict VAS for a file")

    uploaded_file = st.file_uploader(label="Upload your CSV or Excel file", type=['csv','xlsx'])

    global df
    if uploaded_file is not None:
        print("hello")
        print(uploaded_file)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)

        st.write(df)

    algorithms1=("Random Forest",
                    "Bagged CART",
                    "Stacking",
                    "Logistic Regression",
                    "Boosting",)
    algorithms = st.selectbox("Algorithm",algorithms1)

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
   
        #for row in df.iterrows():
        for i in range(len(df)):
            #print(row)
            #print(df.iloc[0])
            #X = np.array([row])
            X = np.array([df.iloc[i]])
            #X = np.array([row])
            print(X)
            X[:, 1] = le_age.transform(X[:,1])
            X[:, 2] = le_conAge.transform(X[:,2])
            X[:, 3] = le_language.transform(X[:,3])
            X[:, 6] = le_device.transform(X[:,6])
    
            #print(X)
            X = X.astype(float)
            #print(X)
            pred = regressor.predict(X)
            #print(X)
            #print(pred)
           # st.write("VAS service11: {}".format(pred[0]))
            prediction = le_package.inverse_transform(pred)
            st.write(prediction[0])

    img = Image.open("Accuracy.jpg")
    st.image(img)

