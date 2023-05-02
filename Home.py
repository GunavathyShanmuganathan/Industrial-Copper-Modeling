import streamlit as st
st.title("Industrial Copper Modeling")
from PIL import Image
image = Image.open(r'C:\Users\User\cop.png')
st.image(image,width=600)
st.write("The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.This web application will save time on data analysis and improves strategic decision making." 
        )
