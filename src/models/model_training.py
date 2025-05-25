import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split

def feature_engineering(X):
    """
    Realiza ingeniería de características para mejorar el rendimiento del modelo.
    
    Args:
        X: DataFrame con las características
        
    Returns:
        DataFrame con características adicionales
    """
    X_eng = X.copy()
    
    # Convertir a numpy para operaciones más eficientes
    X_np = X_eng.values
    
    # Estadísticas básicas
    X_eng['mean_score'] = np.mean(X_np, axis=1)
    X_eng['std_score'] = np.std(X_np, axis=1)
    X_eng['max_score'] = np.max(X_np, axis=1)
    X_eng['min_score'] = np.min(X_np, axis=1)
    X_eng['score_range'] = X_eng['max_score'] - X_eng['min_score']
    
    # Estadísticas adicionales
    X_eng['median_score'] = np.median(X_np, axis=1)
    X_eng['score_variance'] = np.var(X_np, axis=1)
    
    # Calcular skewness y kurtosis manualmente
    mean = X_eng['mean_score'].values
    std = X_eng['std_score'].values
    n = X_np.shape[1]
    
    # Skewness
    skewness = np.zeros(len(X_eng))
    for i in range(len(X_eng)):
        skewness[i] = np.mean(((X_np[i] - mean[i]) / std[i])**3) if std[i] != 0 else 0
    X_eng['score_skew'] = skewness
    
    # Kurtosis
    kurtosis = np.zeros(len(X_eng))
    for i in range(len(X_eng)):
        kurtosis[i] = np.mean(((X_np[i] - mean[i]) / std[i])**4) - 3 if std[i] != 0 else 0
    X_eng['score_kurtosis'] = kurtosis
    
    # Rangos percentiles
    X_eng['p25_score'] = np.percentile(X_np, 25, axis=1)
    X_eng['p75_score'] = np.percentile(X_np, 75, axis=1)
    X_eng['iqr_score'] = X_eng['p75_score'] - X_eng['p25_score']
    
    # Conteo de valores
    mean_values = X_eng['mean_score'].values.reshape(-1, 1)
    X_eng['count_above_mean'] = np.sum(X_np > mean_values, axis=1)
    X_eng['count_below_mean'] = np.sum(X_np < mean_values, axis=1)
    
    return X_eng

def train_model(X_train, y_train):
    """
    Entrena el modelo XGBoost con los parámetros optimizados.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        
    Returns:
        xgb.XGBClassifier: Modelo entrenado
    """
    # Aplicar ingeniería de características
    X_train_eng = feature_engineering(X_train)
    
    # Dividir los datos de entrenamiento en entrenamiento y validación
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_eng, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Definir el espacio de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0.1, 1.0]
    }
    
    # Crear el modelo base
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        random_state=42,
        tree_method='hist',
        enable_categorical=True
    )
    
    # Realizar búsqueda de hiperparámetros
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,  # Número de iteraciones
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("Iniciando búsqueda de hiperparámetros...")
    random_search.fit(X_train_final, y_train_final)
    
    print("\nMejores hiperparámetros encontrados:")
    print(random_search.best_params_)
    
    # Obtener el mejor modelo y sus parámetros
    best_params = random_search.best_params_
    
    # Crear y entrenar el modelo final con early stopping
    final_model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softmax',
        random_state=42,
        tree_method='hist',
        enable_categorical=True,
        early_stopping_rounds=20
    )
    
    print("\nEntrenando modelo final con early stopping...")
    final_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    return final_model, X_train_eng

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y muestra las métricas de rendimiento.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
    """
    # Aplicar ingeniería de características a los datos de prueba
    X_test_eng = feature_engineering(X_test)
    
    y_pred = model.predict(X_test_eng)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MÉTRICAS DE EVALUACIÓN DEL MODELO")
    print("="*50)
    print(f"\nAccuracy (Precisión): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print("\nInterpretación:")
    print("- Accuracy: Porcentaje de predicciones correctas")
    print("- F1-Score: Media armónica entre precisión y recall")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\n" + "="*50)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()
    
    return accuracy, f1, conf_matrix

def plot_feature_importance(model, feature_names):
    """
    Genera un gráfico de importancia de características.
    
    Args:
        model: Modelo entrenado
        feature_names: Nombres de las características
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Importancia de Características')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def save_model_and_transformers(model, scaler, label_encoder, model_dir='../models'):
    """
    Guarda el modelo y los transformadores.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler usado para normalización
        label_encoder: Codificador de etiquetas
        model_dir: Directorio donde guardar los archivos
    """
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'xgboost_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))
    
    print("Modelo y transformadores guardados exitosamente.")

# Ejemplo de uso:
if __name__ == "__main__":
    from data_processing import load_data, preprocess_data, split_data
    
    # Cargar y preprocesar datos
    file_path = '../data/raw/Perfil-Semestre2.xlsx'
    df = load_data(file_path)
    df_processed, scaler, label_encoder = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Entrenar modelo
    model, X_train_eng = train_model(X_train, y_train)
    
    # Evaluar modelo
    evaluate_model(model, X_test, y_test)
    
    # Mostrar importancia de características
    plot_feature_importance(model, X_train_eng.columns)
    
    # Guardar modelo y transformadores
    save_model_and_transformers(model, scaler, label_encoder) 