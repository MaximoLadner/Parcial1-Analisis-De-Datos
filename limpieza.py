import pandas as pd
import numpy as np

# ==============================
# 1️⃣ Cargar dataset
# ==============================
CSV = "base-lesionados-2021.csv"
df = pd.read_csv(CSV, encoding="latin1", sep=";")

# ==============================
# 2️⃣ Mostrar nombres de columnas
# ==============================
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
