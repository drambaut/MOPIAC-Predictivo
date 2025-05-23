import pandas as pd
import numpy as np
import joblib
import os

def load_model_and_transformers():
    """
    Carga el modelo entrenado y los transformadores necesarios.
    """
    model = joblib.load('models/xgboost_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    
    return model, scaler, label_encoder

def preprocess_new_data(df, scaler):
    """
    Preprocesa nuevos datos usando el scaler guardado.
    """
    # Eliminar columna Usuario si existe
    df = df.drop('Usuario', axis=1, errors='ignore')
    
    # Aplicar el mismo escalado que se usó en entrenamiento
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def predict(model, scaler, label_encoder, new_data):
    """
    Realiza predicciones sobre nuevos datos.
    """
    # Preprocesar datos
    processed_data = preprocess_new_data(new_data, scaler)
    
    # Realizar predicciones
    predictions = model.predict(processed_data)
    
    # Convertir predicciones numéricas a etiquetas originales
    original_labels = label_encoder.inverse_transform(predictions)
    
    return original_labels

def main():
    # Verificar que el modelo y los transformadores existan
    required_files = [
        'models/xgboost_model.joblib',
        'models/scaler.joblib',
        'models/label_encoder.joblib'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"El archivo {file} no existe. Asegúrese de haber entrenado el modelo primero.")
    
    # Cargar modelo y transformadores
    model, scaler, label_encoder = load_model_and_transformers()
    
    # Ejemplo de uso con datos nuevos
    # Nota: En un caso real, estos datos vendrían de una fuente externa
    print("Para hacer predicciones, use la función predict() con sus datos.")
    print("Ejemplo de uso:")
    print("""
    # Cargar datos nuevos
    new_data = pd.read_excel('ruta/a/sus/datos.xlsx', skiprows=1)
    
    # Hacer predicciones
    predictions = predict(model, scaler, label_encoder, new_data)
    
    # Ver resultados
    print(predictions)
    """)

if __name__ == "__main__":
    main() 