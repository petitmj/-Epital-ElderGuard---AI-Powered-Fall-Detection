import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    logging.info(f"Total missing values: {missing_values}")
    
    # Check class distribution
    class_counts = df['label'].value_counts()
    logging.info(f"Class Distribution:\n{class_counts}")
    
    # Normalize Data (Standard Scaling)
    scaler = StandardScaler()
    features = df.iloc[:, 1:]  # Exclude label column
    scaled_features = scaler.fit_transform(features)
    
    # Convert back to DataFrame
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[1:])
    df_scaled['label'] = df['label']  # Retain labels
    
    return df_scaled, scaler

def apply_pca(df_scaled):
    # Feature Selection using PCA (Dimensionality Reduction)
    pca = PCA(n_components=0.95)  # Retain 95% variance
    principal_components = pca.fit_transform(df_scaled.iloc[:, :-1])  # Exclude label
    
    # Convert PCA results into a DataFrame
    df_pca = pd.DataFrame(principal_components)
    df_pca['label'] = df_scaled['label']
    
    logging.info(f"Original feature count: {df_scaled.shape[1] - 1}")
    logging.info(f"Reduced feature count after PCA: {df_pca.shape[1] - 1}")
    
    return df_pca, pca

def visualize_class_distribution(class_counts):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Activity Type")
    plt.ylabel("Count")
    plt.show()

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    return accuracy

def save_model_artifacts(model, scaler, label_encoder):
    joblib.dump(model, "fall_detection_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    logging.info("✅ Model and encoders saved successfully!")

def train_hybrid_model(data, labels):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)

        # Isolation Forest
        isolation_model = IsolationForest(contamination=0.05, random_state=42)
        isolation_model.fit(X_train)
        y_pred_iso = isolation_model.predict(X_test)
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

        # Autoencoder
        input_dim = X_train.shape[1]
        autoencoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ])

        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

        reconstructions = autoencoder.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred_auto = (mse > threshold).astype(int)

        # Hybrid Prediction (Majority Voting)
        y_pred_hybrid = (y_pred_iso + y_pred_auto) >= 1
        y_pred_hybrid = y_pred_hybrid.astype(int)

        unique_labels = np.unique(y_test)
        average_mode = "binary" if len(unique_labels) <= 2 else "weighted"

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_hybrid, average=average_mode, zero_division=0
        )

        logging.info(f"Hybrid Model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Save models
        joblib.dump(isolation_model, "isolation_forest.pkl")
        joblib.dump(scaler, "scaler.pkl")
        autoencoder.save("autoencoder.h5")

        return isolation_model, autoencoder, scaler
    except Exception as e:
        logging.error(f"Error training Hybrid Model: {e}")
        return None, None, None

if __name__ == "__main__":
    # Load and preprocess data
    file_path = "fall_detection_dataset.csv"
    df = load_dataset(file_path)
    df_scaled, scaler = preprocess_data(df)
    df_pca, pca = apply_pca(df_scaled)
    df_pca.to_csv("processed_data.csv", index=False)
    logging.info("✅ Preprocessed data saved as 'processed_data.csv'")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df_pca['label'] = label_encoder.fit_transform(df_pca['label'])
    
    # Split dataset
    X = df_pca.drop(columns=["label"])
    y = df_pca["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    save_model_artifacts(rf_model, scaler, label_encoder)
    
   # Train hybrid model
isolation_model, autoencoder, hybrid_scaler = train_hybrid_model(X, y)

if isolation_model is not None:
    logging.info("✅ Hybrid Model training completed!")
else:
    logging.error("❌ Hybrid Model training failed!")