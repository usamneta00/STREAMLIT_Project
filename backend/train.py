import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def train_model():
    print("🚀 Starting model training...")
    
    # Paths
    data_path = os.path.join("backend", "data", "product_recommendation.csv")
    models_dir = os.path.join("backend", "models")
    save_path = os.path.join(models_dir, "logistic_model_outputs.pkl")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    target_col = 'product_name'
    id_col = 'user_id'
    
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Train
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save artifacts
    artifacts = {
        "model": model,
        "label_encoder": label_encoder,
        "attribute_names": X_train.columns.tolist()
    }
    
    joblib.dump(artifacts, save_path)
    print(f"✅ Model trained and saved successfully at {save_path}")

if __name__ == "__main__":
    train_model()
