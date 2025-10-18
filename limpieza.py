import pandas as pd
import numpy as np

# ==============================
# 1️⃣ Cargar dataset
# ==============================
CSV = "base-lesionados-2021.csv"
df = pd.read_csv(CSV, encoding="latin1", sep=";")

# ==============================
# 2️⃣🅰️ Estandarización de nombres de columnas
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


# ======================
# 2️⃣🅱️ Auditoría de calidad
# ======================
# --- Contar nulos por columna ---
# --- Detectar y reemplazar tokens que representan nulos ---
TOKENS_NULOS = { "Sin determinar", "sin determinar"} 
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].replace(list(TOKENS_NULOS), pd.NA)

print("\n❌ Nulos tras reemplazo de tokens")
print(df.isna().sum())


# ==============================
# Reemplazo de valores nulos
# ==============================
## codigo_partido y partido
# Se reemplazaran los valores nulos en esta columna con la Moda (el partido más frecuente)
moda_partido = df["partido"].mode().iloc[0]

# Imputación de 'partido': con la moda general
df["partido"] = df["partido"].fillna(moda_partido)
# Imputación de 'codigo_partido': con el código de el partido correspondiente (La Matanza) 
df["codigo_partido"] = df["codigo_partido"].fillna(427)


## Edad
df["col_edad"] = pd.to_numeric(df["edad"], errors="coerce")
# Se reemplazaran los valores nulos en esta columna con la Mediana
mediana_edad = df["col_edad"].median()

# Imputación de 'edad': con la mediana general
df["edad"] = df["col_edad"].fillna(mediana_edad)
# Eliminación de columnas auxiliares
df = df.drop(columns="col_edad")


## Edad_agrupada
# Se reemplazaran los valores nulos en esta columna con la Moda
moda_edad_agrupada = df['edad_agrupada'].mode().iloc[0]

# Imputación de 'edad_agrupada': con la moda general
df["edad_agrupada"] = df['edad_agrupada'].fillna(moda_edad_agrupada)


## diurno_nocturno
# Se reemplazaran los valores nulos en esta columna con la Moda
moda_d_n = df["diurno_nocturno"].mode().iloc[0]

# Imputación de 'diurno_nocturno': con la moda general
df["diurno_nocturno"] = df["diurno_nocturno"].fillna(moda_d_n)


## sexo
# Se reemplazaran los valores nulos en esta columna con la Moda
moda_sexo = df["sexo"].mode().iloc[0]

# Imputación de 'sexo': con la moda general
df["sexo"] = df["sexo"].fillna(moda_sexo)


print("\n✅ Nulos tras imputacion de datos")
print(df.isna().sum())



# ==============================
# Duplicados
# ==============================

# 1️⃣ Detectar duplicados exactos (todas las columnas)
duplicados_exactos = df.duplicated()
print("\n🟡 Cantidad de filas duplicadas exactas:", duplicados_exactos.sum())

# 2️⃣ Detectar duplicados por columna tipo ID ('codigo_partido')
col_id = "codigo_partido"
if col_id in df.columns:
    duplicados_id = df.duplicated(subset=[col_id])
    print(f"\n🟡 Cantidad de duplicados según columna '{col_id}':", duplicados_id.sum(), "igualmente esto no importa ya que obviamente va a ver registros que compartan el mismo codigo de partido")


# 3️⃣ Eliminar duplicados exactos
filas_antes = df.shape[0]
df = df.drop_duplicates()
filas_despues = df.shape[0]
print(f"\n✅ Se eliminaron {filas_antes - filas_despues} filas duplicadas exactas.")
print(f"Filas restantes en el dataset: {filas_despues} ")


### PUNTO 3 Parcial --------------------------------


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