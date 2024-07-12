import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the model
model = joblib.load('housing_price_model.pkl')

# Load the dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Function to predict housing price
def predict_price(features):
    return model.predict([features])

st.title('Real-Time Housing Price Predictor')

# Sidebar for user input
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    MedInc = st.sidebar.slider('MedInc', float(df.MedInc.min()), float(df.MedInc.max()), float(df.MedInc.mean()))
    HouseAge = st.sidebar.slider('HouseAge', float(df.HouseAge.min()), float(df.HouseAge.max()), float(df.HouseAge.mean()))
    AveRooms = st.sidebar.slider('AveRooms', float(df.AveRooms.min()), float(df.AveRooms.max()), float(df.AveRooms.mean()))
    AveBedrms = st.sidebar.slider('AveBedrms', float(df.AveBedrms.min()), float(df.AveBedrms.max()), float(df.AveBedrms.mean()))
    Population = st.sidebar.slider('Population', float(df.Population.min()), float(df.Population.max()), float(df.Population.mean()))
    AveOccup = st.sidebar.slider('AveOccup', float(df.AveOccup.min()), float(df.AveOccup.max()), float(df.AveOccup.mean()))
    Latitude = st.sidebar.slider('Latitude', float(df.Latitude.min()), float(df.Latitude.max()), float(df.Latitude.mean()))
    Longitude = st.sidebar.slider('Longitude', float(df.Longitude.min()), float(df.Longitude.max()), float(df.Longitude.mean()))
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predict the price
prediction = predict_price(input_df.iloc[0].tolist())
st.write(f'Predicted Price: ${prediction[0]*1000:.2f}')

# Display various graphs
st.subheader('Data Overview')
st.write(df.describe())

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, ax=ax)
st.pyplot(fig)

st.subheader('Distribution of Prices')
fig, ax = plt.subplots()
sns.histplot(df['PRICE'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader('Scatter Plots')
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
for feature in features:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature], y=df['PRICE'], ax=ax)
    ax.set_title(f'Price vs {feature}')
    st.pyplot(fig)