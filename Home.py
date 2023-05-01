import streamlit as st
st.title("Industrial Copper Modeling")
from PIL import Image

image = Image.open(r'C:\Users\User\img.webp')

st.image(image,width=600)
