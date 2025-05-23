import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os
import json

def load_processed_data():
    """
    Carga los datos procesados de entrenamiento y prueba.
    """
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train_data.drop('Perfil', axis=1)
    y_train = train_data['Perfil']
    X_test = test_data.drop('Perfil', axis=1)
    y_test = test_data['Perfil']
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Entrena el modelo XGBoost con los parámetros optimizados.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        early_stopping_rounds=10,
        verbose=True
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y retorna las métricas de rendimiento.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return metrics

def save_model_and_metrics(model, metrics):
    """
    Guarda el modelo y las métricas de evaluación.
    """
    os.makedirs('models', exist_ok=True)
    
    # Guardar modelo
    joblib.dump(model, 'models/xgboost_model.joblib')
    
    # Guardar métricas
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Entrenar modelo
    print("Entrenando modelo XGBoost...")
    model = train_model(X_train, y_train)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Guardar modelo y métricas
    save_model_and_metrics(model, metrics)
    
    # Imprimir métricas
    print("\nMétricas de evaluación:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nMatriz de confusión:")
    print(np.array(metrics['confusion_matrix']))

if __name__ == "__main__":
    main() 