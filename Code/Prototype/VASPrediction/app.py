import streamlit as st
from predict_page import show_predictPage
from predictList_page import show_predictListPage
#show_predictListPage()

page = st.sidebar.selectbox("Predict Single record or list",("Single","List"))

if page == "Single":
    show_predictPage()
else:
    show_predictListPage()