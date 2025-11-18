import streamlit as st
import matplotlib.pyplot as plt
from numpy.random import default_rng as rng

st.set_page_config(page_title='Iris Classifier App', layout='centered')

arr = rng(0).normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)