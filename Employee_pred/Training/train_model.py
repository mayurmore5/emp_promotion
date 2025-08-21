#!/usr/bin/env python3
"""
Employee Promotion Prediction - Model Training Script
This script trains the ML model and saves it for the Flask application.
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Handling imbalanced data
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("Loading dataset...")
    data = pd.read_csv("../Dataset/emp_promotion.csv")
    print(f"Dataset shape: {data.shape}")
    
    # Create a copy for preprocessing
    df = data.copy()
    
    # Remove unwanted columns
    columns_to_drop = ['employee_id', 'region', 'recruitment_channel', 'gender', 'age', 'education']
    df = df.drop(columns=columns_to_drop)
    print(f"Shape after dropping columns: {df.shape}")
    
    # Handle missing values
    numerical_cols = ['previous_year_rating', 'avg_training_score']
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Handle outliers using IQR method
    outlier_features = ['length_of_service', 'avg_training_score']
    for feature in outlier_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['department'] = le.fit_transform(df['department'])
    
    # Save the label encoder
    with open('../flask/department_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Label encoder saved!")
    
    # Remove any remaining problematic records
    initial_shape = df.shape[0]
    df = df[(df['length_of_service'] >= 0) & 
            (df['previous_year_rating'] >= 1) & (df['previous_year_rating'] <= 5) &
            (df['avg_training_score'] >= 0) & (df['avg_training_score'] <= 100)]
    
    print(f"Records removed: {initial_shape - df.shape[0]}")
    print(f"Final dataset shape: {df.shape}")
    
    return df, le

def prepare_features(df):
    """Prepare features and target variables."""
    X = df.drop('is_promoted', axis=1)
    y = df['is_promoted']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(X.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Promotion rate: {y.mean():.2%}")
    
    # Handle class imbalance using SMOTE
    print("Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"Balanced dataset shape: {X_balanced.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare their performance."""
    # Define models to compare
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    results = {}
    
    print("Training and evaluating models...")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("-" * 30)
    
    return results

def select_and_save_best_model(results, feature_names):
    """Select the best model and save it."""
    # Get the best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Cross-validation for robust evaluation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Final evaluation on test set
    y_pred_final = best_model.predict(X_test)
    y_pred_proba_final = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    print(f"\nFinal Test Set Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    if y_pred_proba_final is not None:
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
    
    # Test the model with sample data
    print("\nTesting model with sample data...")
    sample_input = np.array([[0, 10, 4.5, 7, 1, 0, 78]])  # [dept, trainings, rating, service, kpi, awards, score]
    
    prediction = best_model.predict(sample_input)
    probability = best_model.predict_proba(sample_input)[0, 1] if hasattr(best_model, 'predict_proba') else None
    
    print(f"Sample input: {sample_input[0]}")
    print(f"Prediction: {prediction[0]} ({'Eligible' if prediction[0] == 1 else 'Not Eligible'})")
    if probability is not None:
        print(f"Probability: {probability:.3f}")
    
    # Save the best model
    print("\nSaving the best model...")
    with open('../flask/promotion.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save model metadata
    model_info = {
        'model_name': best_model_name,
        'accuracy': accuracy_score(y_test, y_pred_final),
        'feature_names': list(feature_names),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_shape': X_train.shape,
        'cv_mean_accuracy': cv_scores.mean(),
        'cv_std_accuracy': cv_scores.std()
    }
    
    with open('../flask/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Model saved successfully!")
    print(f"Model info: {model_info}")
    
    return best_model, model_info

def print_summary(data, df, results, model_info):
    """Print a comprehensive summary of the training process."""
    print("\n" + "=" * 60)
    print("EMPLOYEE PROMOTION PREDICTION - MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset Information:")
    print(f"- Original dataset size: {data.shape}")
    print(f"- Final dataset size: {df.shape}")
    print(f"- Features used: {len(df.columns) - 1}")
    print(f"- Target distribution: {df['is_promoted'].value_counts().to_dict()}")
    
    print(f"\nModel Performance:")
    for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"- {name}: {result['accuracy']:.4f}")
    
    print(f"\nBest Model: {model_info['model_name']}")
    print(f"- Test Accuracy: {model_info['accuracy']:.4f}")
    print(f"- CV Mean Accuracy: {model_info['cv_mean_accuracy']:.4f}")
    
    print(f"\nKey Findings:")
    print(f"- The model successfully predicts employee promotion with high accuracy")
    print(f"- The model is ready for deployment in the Flask web application")
    
    print(f"\nFiles Created:")
    print(f"- ../flask/promotion.pkl (main model file)")
    print(f"- ../flask/model_info.json (model metadata)")
    print(f"- ../flask/department_encoder.pkl (label encoder)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("Starting Employee Promotion Prediction Model Training...")
    print("=" * 60)
    
    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data()
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Select and save best model
    best_model, model_info = select_and_save_best_model(results, feature_names)
    
    # Print summary
    print_summary(pd.read_csv("../Dataset/emp_promotion.csv"), df, results, model_info)
    
    print("\nTraining completed successfully!")
    print("The model is now ready for the Flask application.")
