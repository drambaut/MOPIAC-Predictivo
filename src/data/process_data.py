import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(file_path):
    """
    Carga los datos desde el archivo Excel, saltando la primera fila.
    """
    df = pd.read_excel(file_path, skiprows=1)
    return df

def preprocess_data(df):
    """
    Realiza el preprocesamiento de los datos:
    1. Elimina la columna Usuario
    2. Maneja valores faltantes
    3. Normaliza variables numéricas
    4. Codifica variables categóricas
    """
    # Eliminar columna Usuario
    df = df.drop('Usuario', axis=1, errors='ignore')
    
    # Separar variables numéricas y categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Manejar valores faltantes en variables numéricas
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Guardar el scaler para uso futuro
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Codificar variable objetivo (Perfil)
    if 'Perfil' in df.columns:
        label_encoder = LabelEncoder()
        df['Perfil'] = label_encoder.fit_transform(df['Perfil'])
        joblib.dump(label_encoder, 'models/label_encoder.joblib')
    
    return df

def split_data(df, target_col='Perfil', test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def main():
    # Crear directorios necesarios
    os.makedirs('data/processed', exist_ok=True)
    
    # Cargar datos
    df = load_data('data/raw/Perfil-Semestre2.xlsx')
    
    # Preprocesar datos
    df_processed = preprocess_data(df)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Guardar datos procesados
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    print("Procesamiento de datos completado exitosamente.")

if __name__ == "__main__":
    main() 