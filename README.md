# ⚽ EPL Match Prediction

This project uses machine learning to predict the outcome of English Premier League (EPL) matches.

## 🧠 Project Goals
- Predict match outcomes (Win/Draw/Loss) using historical EPL data
- Explore team form, home/away stats, and other features
- Train classification models and evaluate performance
- Optional: Build an interactive predictor app with Streamlit

## 📁 Structure
- `data/`: Raw and cleaned datasets
- `notebooks/`: EDA and model training
- `app/`: Streamlit app (optional)
- `models/`: Saved model files
- `utils/`: Feature engineering and helper functions

## 🔧 Tech Stack
- Python, pandas, scikit-learn, XGBoost
- Streamlit (optional)
- Matplotlib, seaborn for visualization

## 🚀 Getting Started
```bash
pip install -r requirements.txt
jupyter notebook notebooks/model_training.ipynb
```

## ✅ Future Enhancements
- Add player-level stats from FPL API
- Integrate betting odds for comparison
- Deploy model as a web app
