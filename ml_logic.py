# ml_logic.py (Full from our chats; add new features here)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from scipy.spatial.distance import cosine
import re
# ... (import all libs: torch, shap, etc.)

# Load/Train Model (Do once; save to models/efficiency_model.joblib)
def train_model(df):  # df with 70+ features
    X = df.drop(['Candidate', 'Efficiency_Score'], axis=1)  # Features
    y = df['Efficiency_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    dump(model, 'models/efficiency_model.joblib')
    return model

# Extract Features (70+ as listed; abbreviated—paste full from before)
def extract_features(df, job_skills):
    # Hard Skills (11)
    df['Skills_Overlap'] = ...  # Full impl from code
    # ... (Add all 70: Predictive like Performance_Predictor = (df['Trajectory_Score'] + df['Proj_Impact']) / 2
    # Bias: df['Similar_to_Me_Bias_Flag'] = np.random.uniform(0, 0.1, len(df))  # AIF360 in prod
    # New from research: df['Inferred_Skills_Score'] = df['Skills_Overlap'] + (df['Niche_Skills'] * 0.2)
    return df

# Compute Efficiency Score
def compute_efficiency(df, model_path='models/efficiency_model.joblib'):
    model = load(model_path)
    features = extract_features(df, job_skills=['Python', 'ML'])  # From job desc
    features['Efficiency_Score'] = model.predict(features.drop('Candidate', axis=1))
    return features

# Candidate Swapping
def find_swaps(df, threshold=0.6):
    # Vectorize 70 features (subset top 20 for speed)
    def feature_vector(row): return np.array([row[f] for f in df.columns[2:22]])  # Skip ID/Name
    df['Feature_Vector'] = df.apply(feature_vector, axis=1)
    swaps = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            sim = 1 - cosine(df['Feature_Vector'].iloc[i], df['Feature_Vector'].iloc[j])
            if sim > threshold:
                swaps.append({'Swap': f"{df['Candidate'].iloc[i]} <-> {df['Candidate'].iloc[j]}", 'Parity': sim})
    return swaps[:3]

# Example Usage (for testing)
if __name__ == "__main__":
    # Synthetic data load/train
    data = {...}  # From before
    df = pd.DataFrame(data)
    model = train_model(df)
    scored_df = compute_efficiency(df)
    swaps = find_swaps(scored_df)
    print(scored_df[['Candidate', 'Efficiency_Score']])
