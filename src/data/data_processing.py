import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Carga los datos desde un archivo Excel, saltando la primera fila.
    
    Args:
        file_path (str): Ruta al archivo Excel
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados
    """
    df = pd.read_excel(file_path, skiprows=1)
    return df

def preprocess_data(df):
    """
    Preprocesa los datos para el modelo:
    1. Elimina la columna Usuario
    2. Maneja valores faltantes
    3. Normaliza variables numéricas
    4. Codifica variables categóricas
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos originales
        
    Returns:
        tuple: (DataFrame procesado, scaler, label_encoder)
    """
    df_processed = df.copy()
    
    # Eliminar columna Usuario y cualquier columna que contenga 'Usuario' en su nombre
    columns_to_drop = [col for col in df_processed.columns if 'Usuario' in col]
    if columns_to_drop:
        print(f"Eliminando columnas: {columns_to_drop}")
        df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Separar variables numéricas y categóricas
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
    
    # Manejar valores faltantes
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    # Inicializar label_encoder
    label_encoder = LabelEncoder()
    
    # Codificar variable objetivo si existe
    if 'Perfil' in df_processed.columns:
        df_processed['Perfil'] = label_encoder.fit_transform(df_processed['Perfil'])
    else:
        # Si no existe la columna Perfil, ajustar el label_encoder con valores vacíos
        label_encoder.fit([])
    
    return df_processed, scaler, label_encoder

def split_data(df_processed, target_col='Perfil', test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        df_processed (pandas.DataFrame): DataFrame procesado
        target_col (str): Nombre de la columna objetivo
        test_size (float): Proporción de datos para prueba
        random_state (int): Semilla aleatoria
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_col not in df_processed.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame")
        
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# Ejemplo de uso:
if __name__ == "__main__":
    # Cargar datos
    file_path = '../data/raw/Perfil-Semestre2.xlsx'
    df = load_data(file_path)
    
    # Preprocesar datos
    df_processed, scaler, label_encoder = preprocess_data(df)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    print("Procesamiento de datos completado exitosamente.") 