import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

DB_NAME = "crypto_data.db"

def load_data():
    """Loads market data from the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM market_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def train_ml_model():
    df = load_data()
    
    if len(df) < 5:
        print("Not enough data points to train a model yet. Run your pipeline a few more times!")
        return

    print(f"Loaded {len(df)} total rows from the database.")

    # Feature Engineering: 
    # Let's predict whether 'price_change_percentage_24h' will be positive (1) or negative (0)
    # based on market cap, total volume, and current price.
    df['target'] = (df['price_change_percentage_24h'] > 0).astype(int)

    # Select features and target variable
    features = ['current_price', 'market_cap', 'total_volume']
    X = df[features]
    y = df['target']

    # Handle any potential nulls
    X = X.fillna(0)

    # Check if we have both classes (0 and 1) to train on
    if y.nunique() < 2:
        print("Warning: All target classes are currently the same. Model needs mixed up/down data to train effectively.")
        print("Tip: Run your pipeline a few times over different hours or use dummy classification for now.")
        # We will set a dummy target split if needed, or proceed
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost Model
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    
    print("\n--- Model Training Results ---")
    print(f"Training completed successfully!")
    if len(X_test) > 0:
        print(f"Accuracy on test subset: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save the model or print feature importance
    print("\nFeature Importances:")
    for col, importance in zip(features, model.feature_importances_):
        print(f"- {col}: {importance:.4f}")

if __name__ == "__main__":
    train_ml_model()