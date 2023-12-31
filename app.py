import pandas as pd 
import numpy as np
import pickle 
import joblib
import os
import streamlit as st

def main():
    # title
    st.title('BigMart Sales Prediction')

    Item_Weight = st.text_input('ITEM WEIGHT')
    Item_Fat_Content = st.sidebar.radio('ITEM FAT CONTENT',[1,0])
    Item_Visibility = st.text_input('ITEM VISIBILITY')
    Item_Type = st.text_input('ITEM TYPE')
    Item_MRP = st.text_input('ITEM MRP')
    Outlet_Establishment_Year = st.sidebar.slider('OUTLET ESTABLISHMENT YEAR',min_value=1985, max_value=2010)
    Outlet_Size = st.sidebar.radio('OUTLET SIZE',[0,1,2])
    Outlet_Location_Type = st.sidebar.radio('OUTLET LOCATION TYPE',[0,1,2])
    Outlet_Type = st.sidebar.radio('OUTLET TYPE',[0,1,2,3])


    if st.button('PREDICT'):
        X = np.array([Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type])

        # Load the scaler
        sc = joblib.load(r'sc.sav')

        # Transform the input data
        X_train_std = sc.transform(X.reshape(1, -1))

        # Load the model
        loaded_model = joblib.load(r'lr.sav')

        # Make predictions
        Y_pred = loaded_model.predict(X_train_std)

        # Display the prediction
        st.success(f'Prediction: {float(Y_pred[0])}')

if __name__ == "__main__":
    main()
