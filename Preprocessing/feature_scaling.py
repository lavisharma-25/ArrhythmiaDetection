from sklearn.preprocessing import StandardScaler

def scale_features(x):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(x)