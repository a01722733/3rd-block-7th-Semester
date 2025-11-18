#here we import libraries and we make the filter by region, displaying the region that you want
import streamlit as st
import pandas as pd

df = pd.read_csv("./sellers.csv", encoding='latin1')

regions = df["REGION"].unique()

option = st.selectbox(
    "Filter by region",
    (regions),
)

st.write("All data is now filtered by", option)

st.dataframe(df[df["REGION"] == option])

st.subheader("Graphs")

# graph of sold units

st.dataframe(df[df["REGION"] == option][["ID", "SOLD UNITS"]])
st.bar_chart(df[df["REGION"] == option][["ID", "SOLD UNITS"]], x="ID", y="SOLD UNITS")


# graph of total sales

st.dataframe(df[df["REGION"] == option][["ID", "TOTAL SALES"]])
st.bar_chart(df[df["REGION"] == option][["ID", "TOTAL SALES"]], x="ID", y="TOTAL SALES")


# graph of average sales

st.dataframe(df[df["REGION"] == option][["ID", "SALES AVERAGE"]])
st.bar_chart(df[df["REGION"] == option][["ID", "SALES AVERAGE"]], x="ID", y="SALES AVERAGE")


#here we display data for a specific vendor, choosing the ID you want and displaying its information
st.subheader("Filter by Vendor (ID)")

vendors = df["ID"].unique()

selected_vendor = st.selectbox("Select a Vendor ID:", vendors)

vendor_data = df[df["ID"] == selected_vendor]

st.write("Data for Vendor ID:", selected_vendor)
st.dataframe(vendor_data)