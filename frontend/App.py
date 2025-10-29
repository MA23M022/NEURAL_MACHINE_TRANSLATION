import streamlit as st
import pandas as pd
import requests


# Specify the URL name 
API_URL = r"http://localhost:8000/predict"       # Give your external url to host the model

# Set title of the page
st.title("Bengali to English translation app")

# Set markdown
st.markdown("Enter your details below")

# set the value of the required field
ben_sentence_id = st.number_input("Enter a index number of a bengali language", value = 980)



# Create a button
if st.button("English translation"):
    input_data = {
        "ben_sentence_id" : ben_sentence_id
    }

    try:
        response = requests.post(API_URL, json = input_data)
        if response.status_code == 200:
            result = response.json()
            show_result = f"ben_sentence : {result['ben_sentence']}, actual_eng_sentence : {result['actual_eng_sentence']}, pred_eng_sentence : {result['pred_eng_sentence']}"
            st.success(f"English Translation : **{show_result}**")
        else:
            st.error(f"API error : {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to fastapi server. Make sure it's running on 8000 port")

