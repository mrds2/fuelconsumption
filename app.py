# Base Libraries

import pandas as pd
import time # To delay response

import warnings 
warnings.filterwarnings("ignore") # To supress warnings

import streamlit as st  # UI Module

import joblib # Saved Models Load Module
import pickle # Saved Encodings Load Module

##################### Trained Model files ######################

# Loading Saved model & pre-process object files
ohe = joblib.load('ohe.pkl')
sc = joblib.load("sc.pkl")
model = joblib.load("fuel_linreg.pkl")

# Loading Saved Ordinal Encoded files
with open('Make_encoding.pkl', 'rb') as f:
    make_encoding = pickle.load(f)
    
with open('Transmission_encoding.pkl', 'rb') as f:
    trans_encoding = pickle.load(f)
    
with open("Vehicle Class_encoding.pkl", 'rb') as f:
    vclass_encoding = pickle.load(f)


######### Sample Input Data to Show to the User ###############

data = pd.read_csv("FuelInpData.csv")

######################## Helper functions for Inputs #####################

if 'sbutton' not in st.session_state:
    st.session_state['sbutton'] = False

if 'fbutton' not in st.session_state:
    st.session_state['fbutton'] = False

def switch_sbutton_state():
    st.session_state['sbutton'] = True
    st.session_state['fbutton'] = False

def switch_fbutton_state():
    st.session_state['sbutton'] = False
    st.session_state['fbutton'] = True

###################################### Design of User Interface ################################

st.subheader(":orange[Fuel Consumption of Vehicles:] :blue[in City (L/100km)]", divider=True)
st.write("Fuel consumption measures the amount of fuel a car consumes to go a specific distance. It is expressed in liters per hundred kilometers. Speed, sudden acceleration, and hard braking can all lead to poorer gas mileage.")
colx, coly, colz = st.columns([1,2,1])

with coly:
    st.image("https://www.epa.gov/sites/default/files/2016-02/ymmv_factors_0.png")

st.divider()
st.write("Sample of Data, Predictive Model Trained...")
st.dataframe(data.head())
st.write("Predicitve Model Trained on above input columns to estimate Fuel Consumption.")
st.divider()
st.subheader(":green[Predictive Modeling:]")
st.write("As our prediction value is numerical, in this Project we have trained on multiple regression algorithms,  \nLinear Regression  \nDecision Tree Regression  \nRandom Forest Regressor  \netc..")
st.divider()
st.subheader(":red[Better Performance Model For Prediction]")
st.write("Among above we got better performance for :green[Linear Regression Algorithm].")
st.write("Trained Linear Regression Algorithm Used for Predictions.")
st.divider()
st.subheader(":blue[Predictions For Given Data:]")

# Prediction Buttons
cola, colb = st.columns(2)
with cola:
    fbutton = st.button("Prediction for Multiple Vehicles By Uploading file...", on_click=switch_fbutton_state, icon=':material/table:')
with colb:
    sbutton = st.button("Prediction for Single Vehicle by Entering Data.....", on_click=switch_sbutton_state, icon=':material/input:')

# Conditions for Predictions Based on Selection
if st.session_state['fbutton'] == True:
    file = st.file_uploader(":red[Upload Test Data File Having X Cols Shown Above:]", type=['csv','xlsx'])
    if file!=None:
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_excel(file)

        st.write(":green[Uploaded Data....]")
        st.dataframe(df.head())

        if st.button(":red[Predict]"):

            # Taking Copy of Uploaded Data
            data = df.copy()

            # Converting text columns to lower case
            for col in data.select_dtypes("O").columns:
                data[col] = data[col].str.lower()

            ############### Using above saved encoded files transforming text columns to numeric ###################

            # Ordinal Encoding
            data['Make'].replace(make_encoding, inplace=True)
            data['Vehicle Class'].replace(vclass_encoding, inplace=True)
            data['Transmission'].replace(trans_encoding, inplace=True)

            # One-Hot Encoding
            data_ohe = ohe.transform(data[['Fuel Type']]).toarray()
            data_ohe = pd.DataFrame(data_ohe, columns=ohe.get_feature_names_out())

            data = pd.concat([data.drop('Fuel Type', axis=1), data_ohe], axis=1)

            # Scaling
            data.iloc[:, 0:8] = sc.transform(data.iloc[:, 0:8])
            
            with st.spinner('Estimating...'):
                # Predictions
                ypred = [round(val,2) for val in model.predict(data)]
                time.sleep(2)
                st.success(":green[Done!]")

                # Taking output column & adding predictions
                df['Fuel Consumption (City (L/100 km)'] = ypred

                st.write(":blue[Predicted Fuel Consumptions....]")
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                st.download_button(label="Download Above Predictions as CSV",data=csv,file_name="predictions.csv",mime="text/csv")

if st.session_state['sbutton'] == True:
    st.write(":red[Enter Details of a Vehicle:]")

    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Select Make:", data.Make.unique())
    with col2:
        vclass = st.selectbox("Select Vehicle Class:", data['Vehicle Class'].unique())

    col3, col4 = st.columns(2)
    with col3:
        esize = st.number_input(f"Enter Engine Size: {data['Engine Size(L)'].min()} to {data['Engine Size(L)'].max()}:")
    with col4:
        cylinders = st.number_input(f"Enter Number of Cylinders: {data['Cylinders'].min()} to {data['Cylinders'].max()}:")

    col5, col6 = st.columns(2)
    with col5:
        trans = st.selectbox("Select Transmission:", data.Transmission.unique())
    with col6:
        fueltype = st.selectbox("Select Fuel Type:", data['Fuel Type'].unique())

    col7, col8, col9 = st.columns(3)
    with col7:
        co2 = st.number_input(f"Enter CO2 Emissions (g/km): {data['CO2 Emissions(g/km)'].min()} to {data['CO2 Emissions(g/km)'].max()}:")
    with col8:
        co2r = st.number_input(f"Enter CO2 Rating: {data['CO2 Rating'].min()} to {data['CO2 Rating'].max()}:")
    with col9:
        smogr = st.number_input(f"Enter Smog Rating: {data['Smog Rating'].min()} to {data['Smog Rating'].max()}:")

    if st.button("Estimate"):

        row = pd.DataFrame([[make,vclass,esize,cylinders,trans,fueltype,co2,co2r,smogr]], columns=data.columns)

        st.write(":green[Given Vehicle Input Data:]")

        st.dataframe(row)

        # Feature Engineering: Need to apply same steps done for training, while giving it to model for prediction

        # Ordinal Encoding
        row['Make'].replace(make_encoding, inplace=True)
        row['Vehicle Class'].replace(vclass_encoding, inplace=True)
        row['Transmission'].replace(trans_encoding, inplace=True)

        # One-Hot Encoding
        row_ohe = ohe.transform(row[['Fuel Type']]).toarray()
        row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())

        row = pd.concat([row.drop('Fuel Type', axis=1), row_ohe], axis=1)

        # Scaling
        row.iloc[:, 0:8] = sc.transform(row.iloc[:, 0:8])

        # Prediction
        fuelconsumption = round(model.predict(row)[0],2)
        st.write(f":blue[Estimated Fuel Consumption (City (L/100Km)):] {fuelconsumption}")

