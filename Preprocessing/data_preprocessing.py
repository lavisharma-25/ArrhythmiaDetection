# import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the dataset: label encode categorical variables and separate features and target.
    """
    categorical_columns = ['Gender', 'Cholesterol_levels', 'Stress_level', 'Physical_activity']
    label_encoders = {}

    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    x = df.drop('Arrhythmia', axis=1)
    y = df['Arrhythmia']
    return x, y