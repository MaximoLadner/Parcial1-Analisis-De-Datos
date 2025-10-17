import pandas as pd
import seaborn as sns
import numpy as np

# ==============================
# Cargar dataset
# ==============================
df = pd.read_csv("base-lesionados-2021.csv", encoding="latin1")




# ---------------------------------------------
# Detectar y tratar valores nulos
# ---------------------------------------------
print("\nCantidad de nulos por columna:")
print(df.isna().sum())

# Reemplazar nulos en edad (si los hay)
if "edad" in df.columns and df["edad"].isna().sum() > 0:
    df["edad"].fillna(df["edad"].mean(), inplace=True)


# ---------------------------------------------
# Duplicados
# ---------------------------------------------
dup = df.duplicated().sum()
print(f"\nFilas duplicadas encontradas: {dup}")
df.drop_duplicates(inplace=True)

# ---------------------------------------------
# Tipos de datos
# ---------------------------------------------
print("\nTipos de datos finales:")
print(df.dtypes)
