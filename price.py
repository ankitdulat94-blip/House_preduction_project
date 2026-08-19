import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

st.title("🏠 House Price Prediction")

# Dataset
data = {
    "area": [1000,1200,1500,1800,2000,2200,2400,2800,3000,3500],
    "bedrooms": [2,2,3,3,3,4,4,4,5,5],
    "bathrooms": [1,2,2,2,3,3,3,4,4,5],
    "age": [10,8,5,7,6,4,3,5,2,1],
    "price": [40,48,60,70,75,90,100,115,130,150]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["area", "bedrooms", "bathrooms", "age"]]
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = Sequential([
    Dense(64, activation="relu", input_shape=(4,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# Train Model
with st.spinner("Training Model..."):
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=2,
        validation_split=0.2,
        verbose=0
    )

loss, mae = model.evaluate(X_test, y_test, verbose=0)

st.success(f"Model Trained Successfully! Test MAE: {mae:.2f}")

st.subheader("Enter House Details")

area = st.number_input("Area (sq ft)", min_value=500, value=1600)
bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
age = st.number_input("Age of House", min_value=0, value=4)

if st.button("Predict Price"):
    new_house = [[area, bedrooms, bathrooms, age]]
    new_house_scaled = scaler.transform(new_house)

    prediction = model.predict(new_house_scaled, verbose=0)

    st.success(f"Predicted House Price: ₹ {prediction[0][0]:.2f} Lakhs")