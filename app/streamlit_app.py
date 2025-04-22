import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("⚽ EPL Match Outcome Predictor")

# Load model and scaler
model = pickle.load(open("models/xgboost_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# User inputs
st.subheader("Enter Match Stats:")
home_possession = st.slider("Home Possession (%)", 20, 80, 50)
away_possession = 100 - home_possession
home_shots = st.slider("Home Shots", 0, 30, 10)
away_shots = st.slider("Away Shots", 0, 30, 10)
home_corners = st.slider("Home Corners", 0, 15, 5)
away_corners = st.slider("Away Corners", 0, 15, 5)

# Assemble input
X_input = pd.DataFrame([[
    home_possession, away_possession,
    home_shots, away_shots,
    home_corners, away_corners
]], columns=[
    "home_possession", "away_possession",
    "home_shots", "away_shots",
    "home_corners", "away_corners"
])

# Predict
X_scaled = scaler.transform(X_input)
pred = model.predict(X_scaled)[0]
label_map = {0: "Loss", 1: "Draw", 2: "Win"}
st.success(f"🏁 Predicted Match Result (Home Team): **{label_map[pred]}**")
