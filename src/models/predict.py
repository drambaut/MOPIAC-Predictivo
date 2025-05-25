import pandas as pd
import numpy as np
import joblib
import os

def load_model_and_transformers(model_dir='../models'):
    """
    Carga el modelo entrenado y los transformadores necesarios.
    
    Args:
        model_dir: Directorio donde están guardados los archivos
        
    Returns:
        tuple: (modelo, scaler, label_encoder)
    """
    model = joblib.load(os.path.join(model_dir, 'xgboost_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    
    return model, scaler, label_encoder

def preprocess_new_data(df, scaler):
    """
    Preprocesa nuevos datos usando el scaler guardado.
    
    Args:
        df: DataFrame con nuevos datos
        scaler: Scaler usado para normalización
        
    Returns:
        pandas.DataFrame: Datos preprocesados
    """
    df_processed = df.copy()
    
    # Eliminar columna Usuario si existe
    df_processed = df_processed.drop('Usuario', axis=1, errors='ignore')
    
    # Aplicar el mismo escalado
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
    
    return df_processed

def predict(model, scaler, label_encoder, new_data):
    """
    Realiza predicciones sobre nuevos datos.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler usado para normalización
        label_encoder: Codificador de etiquetas
        new_data: DataFrame con nuevos datos
        
    Returns:
        numpy.ndarray: Predicciones
    """
    # Preprocesar datos
    processed_data = preprocess_new_data(new_data, scaler)
    
    # Realizar predicciones
    predictions = model.predict(processed_data)
    
    # Convertir predicciones numéricas a etiquetas originales
    original_labels = label_encoder.inverse_transform(predictions)
    
    return original_labels

# Ejemplo de uso:
if __name__ == "__main__":
    # Cargar modelo y transformadores
    model, scaler, label_encoder = load_model_and_transformers()
    
    # Ejemplo de uso con datos nuevos
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