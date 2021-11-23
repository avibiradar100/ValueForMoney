import streamlit as st
import pickle
import numpy as np

# page title

st.set_page_config(page_title="Laptop Price Predictor",page_icon="💻")

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")
st.sidebar.title("Enter specification")
# brand
company = st.sidebar.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.sidebar.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.sidebar.selectbox('RAM(in GB)',df['Ram'].unique())

# weight
weight = st.sidebar.selectbox('Weight of the Laptop(kg)',[1.0,1.5,1.75,2.0,2.5,3,3.5])

# Touchscreen
touchscreen = st.sidebar.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.sidebar.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.sidebar.selectbox('Screen Size',[14,14.5,15,15.6,16,17,17.3])

# resolution
resolution = st.sidebar.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.sidebar.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.sidebar.selectbox('HDD(in GB)',[500,1000,2000])

ssd = st.sidebar.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.sidebar.selectbox('GPU',df['Gpu brand'].unique())

os = st.sidebar.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res*2) + (Y_res*2))*0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))