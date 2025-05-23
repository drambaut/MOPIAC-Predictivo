# MOPIAC-Predictivo

Proyecto de análisis predictivo para determinar el perfil de estudiantes basado en variables académicas.

## Estructura del Proyecto

```
.
├── data/                   # Directorio para datos
│   ├── raw/               # Datos originales
│   └── processed/         # Datos procesados
├── notebooks/             # Jupyter notebooks para análisis
├── src/                   # Código fuente
│   ├── data/             # Scripts de procesamiento de datos
│   ├── features/         # Scripts de ingeniería de características
│   ├── models/           # Scripts de entrenamiento y evaluación
│   └── visualization/    # Scripts de visualización
├── models/               # Modelos entrenados
└── requirements.txt      # Dependencias del proyecto
```

## Descripción

Este proyecto implementa un modelo predictivo para determinar el perfil de estudiantes basado en variables académicas. El modelo utiliza XGBoost y está diseñado para ser reproducible y desplegable.

### Características principales

- Análisis exploratorio de datos (EDA)
- Preprocesamiento automático de datos
- Entrenamiento de modelo XGBoost
- Evaluación de rendimiento
- Pipeline reproducible

## Instalación

1. Clonar el repositorio
2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Ejecutar el notebook de EDA:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```
2. Ejecutar el pipeline de entrenamiento:
   ```bash
   python src/models/train_model.py
   ```

## Próximos Pasos

- [ ] Implementar API REST para predicciones
- [ ] Desarrollar dashboard de monitoreo
- [ ] Configurar CI/CD para despliegue automático
- [ ] Documentar API y endpoints