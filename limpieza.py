import pandas as pd
import numpy as np

# ==============================
# 1️⃣ Cargar dataset
# ==============================
CSV = "base-lesionados-2021.csv"
df = pd.read_csv(CSV, encoding="latin1", sep=";")

# ==============================
# 2️⃣ Estandarización de nombres de columnas
# ==============================
nuevas_columnas = {
    'CÓDIGO DE PARTIDO (SEGÚN CODIFICACIÓN INDEC)': 'codigo_partido',
    'PARTIDO': 'partido',
    'FECHA': 'fecha',
    'MES': 'mes',
    'DÍA DE LA SEMANA': 'dia_semana',
    'DÍA DE LA SEMANA AGRUPADO': 'dia_semana_agrupado',
    'HORA': 'hora',
    'DIURNO / NOCTURNO': 'diurno_nocturno',
    'EDAD': 'edad',
    'EDAD AGRUPADA': 'edad_agrupada',
    'SEXO': 'sexo'
}
df.columns = [col.upper() if col.upper() not in nuevas_columnas else nuevas_columnas[col.upper()] for col in df.columns]

# Mostrar nombres de columnas
print("🧩 Nombres de columnas del dataset:")
for col in df.columns:
    print(f"- {col}")


# ==============================
# 3️⃣ Buscar automáticamente columna de edad
# ==============================
col_edad = None
for c in df.columns:
    if "edad" in c.lower():
        col_edad = c
        break

if col_edad:

    # Asegurarse de que sea numérica
    df[col_edad] = pd.to_numeric(df[col_edad], errors="coerce")

    print("\n📊 Estadísticos originales de 'edad':")
    print(df[col_edad].describe()[["mean", "50%", "std"]])

    # ==============================
    # 4️⃣ Imputación con mediana
    # ==============================
    df_mediana = df.copy()
    df_mediana[col_edad] = df_mediana[col_edad].fillna(df_mediana[col_edad].median())

    print("\n🧮 Estadísticos después de imputar con la mediana:")
    print(df_mediana[col_edad].describe()[["mean", "50%", "std"]])

    # ==============================
    # 5️⃣ Imputación por grupo (por ejemplo, por sexo)
    # ==============================
    col_sexo = None
    for c in df.columns:
        if "sexo" in c.lower():
            col_sexo = c
            break

    if col_sexo:
        df_grupo = df.copy()
        df_grupo[col_edad] = df_grupo.groupby(col_sexo)[col_edad].transform(
            lambda x: x.fillna(x.mean())
        )

        print(f"\n👥 Estadísticos después de imputar por grupo ('{col_sexo}'):")
        print(df_grupo[col_edad].describe()[["mean", "50%", "std"]])
    else:
        print("\n⚠️ No se encontró la columna 'sexo' para imputar por grupo.")
else:
    print("\n⚠️ No se encontró ninguna columna relacionada con 'edad'.")



# ======================
# 6️⃣ Valores nulos
# ======================
# --- Contar nulos por columna ---
print("\n❌ Nulos por columna")
print(df.isna().sum())
# --- Detectar y reemplazar tokens que representan nulos ---
TOKENS_NULOS = {"", " ", "   ", "-", "na", "NA", "NaN", "nan", "N/A", "Desconocido", "desconocido", "Sin determinar", "sin determinar"} 
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].replace(list(TOKENS_NULOS), pd.NA)

print("\n❌ Nulos tras reemplazo de tokens")
print(df.isna().sum())


# ==============================
# 7️⃣ Reemplazo de valores nulos
# ==============================
df["col_edad"] = pd.to_numeric(df["edad"], errors="coerce")
# Imputación de 'edad': con la mediana general
mediana_edad = df["col_edad"].median()
print(mediana_edad)
df["edad_mediana"] = df["col_edad"].fillna(mediana_edad)
df["edad"] = df["edad_mediana"]

print(df.isna().sum())