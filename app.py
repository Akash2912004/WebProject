import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Your dataset
data = {
    'temperature': [30, 28, 22, 18, 15, 12, 10, 25, 20, 8, 5, 19],
    'humidity': [40, 45, 60, 70, 80, 90, 95, 50, 65, 85, 92, 68],
    'wind_speed': [5, 7, 10, 12, 15, 20, 25, 6, 9, 18, 22, 11],
    'outfit': [
        'light', 'light', 'warm', 'warm', 'rain gear', 'rain gear', 'rain gear',
        'light', 'warm', 'rain gear', 'rain gear', 'warm'
    ]
}
df = pd.DataFrame(data)

# Encode target labels
labels, class_names = pd.factorize(df['outfit'])
X = df[['temperature', 'humidity', 'wind_speed']].values
y = labels

# Train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=1)
dt_model.fit(X, y)

# Streamlit user interface
st.title("Weather-Based Outfit Recommendation")

temp = st.slider("Temperature (°C)", 0, 40, 20)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind = st.slider("Wind Speed (km/h)", 0, 30, 10)

if st.button("Predict Outfit"):
    user_input = np.array([[temp, humidity, wind]])
    prediction = dt_model.predict(user_input)
    st.success(f"Recommended outfit: **{class_names[prediction[0]]}**")
