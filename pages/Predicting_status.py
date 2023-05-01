import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re

st.set_page_config(layout="wide")
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://www.dreamstime.com/yellow-textured-plain-background-wallpaper-design-use-text-image-yellow-textured-plain-background-wallpaper-image137515345");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True) 
setting_bg()

st.write("""
<div style='text-align:center'>
    <h1 style='color:#FF7F50;'>Industrial Copper Modeling</h1>
</div>
""", unsafe_allow_html=True)
st.write("""
<div style='text-align:center'>
    <h1 style='color:#FF7F50;'>Predicting Status</h1>
</div>
""", unsafe_allow_html=True)

# Define the possible values for the dropdown menus
status_op = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
             'Offerable']
item_type_op = ['PL', 'W', 'IPL', 'WI', 'Others', 'S']
country_op = [28, 25, 30, 32, 38, 78, 27, 77, 113, 79, 26, 39, 40, 84, 80, 107, 89]
application_op = [10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 66, 29, 22, 40, 25., 67., 79., 3., 99., 2., 5.,
                  39., 69., 70., 65., 58., 68.]
product_op = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
              '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
              '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
              '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
              '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

with st.form("my_form1"):
    
    cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
    cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
    cwidth = st.text_input("Enter width (Min:1, Max:2990)")
    ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
    cselling = st.text_input("Selling Price (Min:1, Max:100001015)")
    citem_type = st.selectbox("Item Type", item_type_op, key=21)
    ccountry = st.selectbox("Country", sorted(country_op), key=31)
    capplication = st.selectbox("Application", sorted(application_op), key=41)
    cproduct_ref = st.selectbox("Product Reference", product_op, key=51)
    csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #FF7F50;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    cflag = 0
    pattern = "^(?:\d+|\d*\.\d+)$"
    for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
        if re.match(pattern, k):
            pass
        else:
            cflag = 1
            break

if csubmit_button and cflag == 1:
    if len(k) == 0:
        st.write("please enter a valid number space not allowed")
    else:
        st.write("You have entered an invalid value: ", k)

if csubmit_button and cflag == 0:
    import pickle

    with open(r"C:\Users\User\Industrial_Copper_Modelling\source\cmodel.pkl", 'rb') as file:
        cloaded_model = pickle.load(file)

    with open(r'C:\Users\User\Industrial_Copper_Modelling\source\cscaler.pkl', 'rb') as f:
        cscaler_loaded = pickle.load(f)

    with open(r"C:\Users\User\Industrial_Copper_Modelling\source\ccategorical.pkl", 'rb') as f:
        ccategorical_loaded = pickle.load(f)

    new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                            np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(cproduct_ref),
                            citem_type]])
    new_sample_OHE = ccategorical_loaded.transform(new_sample[:, [8]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_OHE), axis=1)
    new_sample = cscaler_loaded.transform(new_sample)
    new_pred = cloaded_model.predict(new_sample)
    if new_pred == 1:
        st.write('## :green[The Status is Won] ')
    else:
        st.write('## :red[The status is Lost] ')