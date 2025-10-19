import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    # 5️⃣  Imputación por grupo (por ejemplo, por sexo)
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


# ==============================
#  6️⃣ Derivación de métricas: variaciones porcentuales mensuales
#    La variación porcentual muestra si los casos aumentaron o disminuyeron respecto al mes previo.
#    Numero positivo → aumentó.
#    Numero negativo → bajó.
#    NaN → no hay comparación posible (primer mes).
# ==============================
if "mes" in df.columns:
    # Contar cantidad de registros por mes
    df_mensual = df.groupby("mes").size().reset_index(name="cantidad")
    df_mensual = df_mensual.sort_values("mes")

    # Calcular variación porcentual mes a mes
    df_mensual["variacion_%"] = df_mensual["cantidad"].pct_change() * 100

    print("\n📊 Variación porcentual mensual de registros:")
    print(df_mensual)


# ==============================
# 7️⃣ Histograma simple de una variable numérica (elegimos edad)
# ==============================

if "edad" in df.columns:
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

    plt.figure(figsize=(8, 5))
    plt.hist(df["edad"].dropna(), bins=20, edgecolor="black")
    plt.title("Distribución de edades")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
else:
    print("\n⚠️ No se encontró la columna 'edad' para generar el histograma.")



# ==============================
# 8️⃣ Histograma superpuesto de dos variables comparables (elegimos edad y sexo)
# ==============================

# Verificar que existan las columnas necesarias
if "edad" in df.columns and "sexo" in df.columns:
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

    # Dividir el dataset por sexo
    hombres = df[df["sexo"].str.lower() == "masculino"]["edad"].dropna()
    mujeres = df[df["sexo"].str.lower() == "femenino"]["edad"].dropna()

    # Crear histograma superpuesto
    plt.figure(figsize=(8, 5))
    plt.hist(hombres, bins=20, alpha=0.6, label="Masculino", edgecolor="black")
    plt.hist(mujeres, bins=20, alpha=0.6, label="Femenino", edgecolor="black")

    plt.title("Distribución de edad por sexo")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
else:
    print("\n⚠️ No se encontraron las columnas 'edad' y/o 'sexo' para graficar el histograma superpuesto.")



# ==============================
# 9️⃣ Boxplot por categoría
# ==============================
### Se puede identificar si los lesionados de la noche son, en promedio, más jóvenes o si tienen más outliers de edad.

if "edad" in df.columns:
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

# Obtener hora como número (hora entera 0-23) desde cadenas "HH:MM"
if "hora" in df.columns:
    df["hora_ts"] = pd.to_datetime(df["hora"], format="%H:%M", errors="coerce")
    df["hora_num"] = df["hora_ts"].dt.hour
else:
    df["hora_num"] = pd.NA



if "diurno_nocturno" in df.columns and "edad" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="diurno_nocturno", y="edad", data=df.dropna(subset=["edad", "diurno_nocturno"]))
    plt.title("Distribución de Edad de Lesionados según Momento del Día")
    plt.xlabel("Momento del Evento")
    plt.ylabel("Edad (Años)")
    plt.grid(axis="y", alpha=0.4)
    plt.show()
else:
    print("\n⚠️ No se encontraron las columnas necesarias para graficar el boxplot ('diurno_nocturno' y 'edad').")



# ==============================
# Scatterplot bivariado
# ==============================
# coloreado por día de la semana agrupado
### Permite explorar la relación entre el momento exacto del día (hora) y la edad.

requerido = {"hora_num", "edad", "dia_semana_agrupado"}
if requerido.issubset(df.columns):
    df_plot = df.dropna(subset=["hora_num", "edad", "dia_semana_agrupado"])
    if df_plot.empty:
        print("\n⚠️ No hay datos válidos para el scatterplot después de limpiar NaN en 'hora_num'/'edad'/'dia_semana_agrupado'.")
    else:
        n = min(5000, len(df_plot))
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x="hora_num", y="edad", hue="dia_semana_agrupado",
                        data=df_plot.sample(n=n, random_state=42), alpha=0.6, s=20)
        plt.title("Relación entre Hora del Accidente y Edad, por Día de la Semana")
        plt.xlabel("Hora del Día (0-23)")
        plt.ylabel("Edad (Años)")
        plt.xticks(range(0, 24))
        plt.grid(axis="both", alpha=0.3)
        plt.show()
else:
    print("\n⚠️ No se encontraron las columnas necesarias para el scatterplot ('hora_num','edad','dia_semana_agrupado').")



# ==============================
# Heatmap de correlaciones
# ==============================
### Permite identificar rápidamente si las personas de mayor edad tienden a accidentarse en horas específicas, por ejemplo.

if "mes" in df.columns and "numero_mes" not in df.columns:
    meses_map = {
        'enero':1, 'febrero':2, 'marzo':3, 'abril':4, 'mayo':5, 'junio':6,
        'julio':7, 'agosto':8, 'septiembre':9, 'octubre':10, 'noviembre':11, 'diciembre':12
    }
    df["numero_mes"] = df["mes"].astype(str).str.strip().str.lower().map(meses_map)

corr_vars = ['edad', 'hora_num', 'numero_mes']
corr_existen = [c for c in corr_vars if c in df.columns]
if len(corr_existen) >= 2:
    correlacion_matriz = df[corr_existen].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(correlacion_matriz, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
    plt.title('Heatmap de Correlaciones entre Variables Numéricas Clave')
    plt.show()
else:
    print(f"\n⚠️ No hay suficientes columnas numéricas disponibles para el heatmap. Columnas encontradas: {corr_existen}")