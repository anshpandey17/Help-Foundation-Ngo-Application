
import streamlit as st
import numpy as np
import pandas as pd
import joblib

#First lets load the instances that we created.

with open('scaler.joblib','rb') as file:
    scale = joblib.load(file)

with open('pca.joblib','rb') as file:
    pca = joblib.load(file)
    
with open('final_model.joblib','rb') as file:
    model = joblib.load(file)

# ____________________________________________________


def prediction(input_list):

    scaled_input = scale.transform([input_list])
    pca_input    = pca.transform(scaled_input)
    output       = model.predict(pca_input)[0]

    if output==0:
        return 'DEVELOPED'
    elif output==1:
        return 'UNDER_DEVELOPED'
    elif:
        return 'DEVELOPING'

# _____________________________________________________


def main():
    st.title('HELP NGO FOUNDATION')
    st.subheader('This Application will give the Status of The Country based on Socio-economic And Health Factors')

    gdp = st.text_input('Enter the GDP per population of a country')
    inc = st.text_input('Enter the Per Capita Income of the country')
    imp = st.text_input('Enter the Imports in terms of GDP of the country')
    exp = st.text_input('Enter the Exports in terms of GDP of the country')
    inf = st.text_input('Enter the Inflation Rate (%) of the country')

    hel = st.text_input('Enter the Expenditure on Health in terms of % of GDP')
    ch_m = st.text_input('Enter the no. of Deaths per 1000 births for <5 years')
    fer = st.text_input('Enter the Avg. Children born to a women in a country')
    lf = st.text_input('Enter the Avg. Life Expectancy in a country')

    in_data = [ch_m,exp,hel,imp,inc,inf,lf,fer,gdp]

    if st.button('Predict'):
        response = prediction(in_data)
        st.success(response)

if __name__=='__main__':
    main()
