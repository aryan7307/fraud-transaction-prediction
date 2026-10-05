#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model3.pkl")
scaler = joblib.load("scaler3.pkl")

st.set_page_config(page_title="fradulent prediction", page_icon="🩺")

st.title("fradulent prediction")

st.write("Enter the feature values below to predict the fradulent.")

# User Inputs
avg_amount_days = st.number_input("Average amount days", min_value=0.0, format="%.4f")
transaction_amount = st.number_input("Transaction ammount", min_value=0.0, format="%.4f")
is_declined = st.number_input("Declined", min_value=0.0, format="%.4f")
number_declines_days = st.number_input("no. declined days", min_value=0.0, format="%.4f")
foreign_transaction = st.number_input("forign transaction", min_value=0.0, format="%.4f")
high_risk_countries = st.number_input("high risk countries", min_value=0.0, format="%.4f")
daily_chbk_avg_amt = st.number_input("daily chbk amt", min_value=0.0, format="%.4f")
six_m_avg_chbk_amt = st.number_input("m avg cgbk amt", min_value=0.0, format="%.4f")
six_m_chbk_freq = st.number_input("m chbk frequency", min_value=0.0, format="%.4f")
if st.button("predict fradulent"):

    input_data = np.array([[avg_amount_days, transaction_amount, is_declined, number_declines_days, foreign_transaction, high_risk_countries, daily_chbk_avg_amt, six_m_avg_chbk_amt, six_m_chbk_freq]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Display Result
    if prediction[0] == "fraudulent" or prediction[0] == 0:
        st.error("🔴    ")
    else:
        st.success("🟢    ")



# In[ ]:




