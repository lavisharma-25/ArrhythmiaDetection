import os
import pandas as pd
from Preprocessing.data_preprocessing import preprocess_data
from Preprocessing.feature_scaling import scale_features
from Visualization.performance_graphs import plot_performance
from Visualization.confusion_matrices import plot_confusion_matrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create Output directory if it doesn't exist
output_dir = 'Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
df = pd.read_csv('arrhythmia_detection_dataset.csv')

# Preprocess the data
X, y = preprocess_data(df)

# Scale features
X_scaled = scale_features(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and evaluate models
performance = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}
cm_list = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Store performance metrics
    performance['Model'].append(model_name)
    performance['Accuracy'].append(accuracy_score(y_test, y_pred))
    performance['Precision'].append(precision_score(y_test, y_pred))
    performance['Recall'].append(recall_score(y_test, y_pred))
    performance['F1-Score'].append(f1_score(y_test, y_pred))

    # Store confusion matrix
    cm_list.append((model_name, confusion_matrix(y_test, y_pred)))

# Plot and save performance comparison
plot_performance(performance, output_dir)

# Plot and save confusion matrices
plot_confusion_matrices(cm_list, models, output_dir)

print("\nModel Performance Comparison:")
print(pd.DataFrame(performance))
