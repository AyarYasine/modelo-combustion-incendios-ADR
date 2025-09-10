import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import chart_studio.plotly as py
import os

from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_ivp
from scipy.stats import linregress
from plotly.subplots import make_subplots

# Log In API Plotly - Esto es para poder guardar los graficos en servidores de plotly y acceder a ellos con URLs publicas. Se deberá crear una cuenta en Plotly.com
# No es obligatorio, se puede ejecutar el programa sin esto. En ese caso, las gráficas se guardaran en .html en el directorio local
py.sign_in('USUARIO', 'CONTRASEÑA API')

# Constantes globales
R = 8.3144              # Constante de los gases en J/mol·K

# Diccionario GLOBAL
DATOS_PROCESADOS = {
    "DatosComponentes": {},
    "DatosMuestras":{},
}

RESULTADOS_GLOBALES = {
    "FWO_Starink_Friedman": {},      # RESULTADOS_GLOBALES["FWO_Starink_Friedman"]["celulosa"]["aire"]["resultados"], esto nos dará los valores que queramos.
}


# MODELOS CINÉTICOS
# Tener en cuenta que las reacciones de orden surgen de la ecuacion de Šesták–Berggren (Mecanismo Autocatalítico General de Šesták–Berggren)
# Funciones f(alpha) para los distintos modelos cinéticos
# Chemical reaction model
def cr0_f(alpha): return 1                                                      # Zero order (CR0)
def cr05_f(alpha): return (1 - alpha)**0.5
def cr1_f(alpha): return 1 - alpha                                              # First order (CR1)
def cr15_f(alpha): return (1 - alpha)**1.5
def cr2_f(alpha): return (1 - alpha)**2                                         # Second order (CR2)
def cr25_f(alpha): return (1 - alpha)**2.5
def cr3_f(alpha): return (1 - alpha)**3                                         # Third order (CR3)
def cr35_f(alpha): return (1 - alpha)**3.5
def cr4_f(alpha): return (1 - alpha)**4

# Diffusion model
def dm1_f(alpha): return 1/(2*alpha)                                            # Parabolic law (1D Diffusion) (DM1)
def dm2_f(alpha): return -(np.log(1 - alpha))**(-1)                             # Valensi (2D diffusion) (DM2)
def dm3_f(alpha): return (3*(1-alpha)**(2/3))/(2*(1-(1-alpha)**(1/3)))          # Jander (3D Diffusion)
def dm4_f(alpha): return 3/(2*((1 - alpha)**(-1/3) - 1))                        # Ginstling - Broushtein (3D Diffusion) (DM3)
def dm5_f(alpha): return (3/2)*(1-alpha)**(4/3)*((1-alpha)**(-1/3) - 1)**(-1)   # Zhuravlev-Lesokhin-Tempelman

# Nucleation and growth model
def ng15_f(alpha): return 1.5*(1-alpha)*(-np.log(1 - alpha))**(1/3)           # Avrami - Erofeev (n = 1.5) (NG1.5)
def ng2_f(alpha): return 2*(1-alpha)*(-np.log(1 - alpha))**(1/2)                # Avrami - Erofeev (n = 2) (NG2)
def ng3_f(alpha): return 3*(1-alpha)*(-np.log(1 - alpha))**(2/3)                # Avrami - Erofeev (n = 3) (NG3)
def ng4_f(alpha): return 4*(1-alpha)*(-np.log(1 - alpha))**(3/4)                # Avrami - Erofeev (n = 4) (NG4)
def b1_f(alpha): return alpha*(1-alpha)                                         # Prout-Tompkins
def p23_f(alpha): return 2/(3*alpha**-0.5)                                      # Power law
def p2_f(alpha): return 2*alpha**(1/2)                                          # Power law
def p3_f(alpha): return 3*alpha**(2/3)                                          # Power law
def p4_f(alpha): return 4*alpha**(3/4)                                          # Power law
def e1_f(alpha): return alpha                                                   # Exponent Power first-order
def e2_f(alpha): return 0.5*alpha                                               # Exponent Power second-order

# Geometrical contraction models
def r2_f(alpha): return 2**(1-alpha)**(1/3)                                     # Contracting cylinder
def r3_f(alpha): return 1 - (1-alpha)**(2/3)                                    # Contracting volume


# Funciones g(alpha) para los distintos modelos cinéticos
# Chemical reaction model
def cr0_g(alpha): return alpha                                                  # Zero order (CR0)
def cr05_g(alpha): return 2*(1-(1-alpha)**(1/2))
def cr1_g(alpha): return -np.log(1 - alpha)                                     # First order (CR1)
def cr15_g (alpha): return 2*((1-alpha)**(-1/2) - 1)
def cr2_g(alpha): return 2 * ((1 - alpha) ** (-1.5) - 1)                        # Second order (CR2)
def cr25_g(alpha): return (2/3)*((1-alpha)**(-3/2) - 1)
def cr3_g(alpha): return 0.5 * ((1 - alpha) ** (-2) - 1)                        # Third order (CR3)
def cr35_g(alpha): return (2/5)*((1-alpha)**(-5/2) - 1)
def cr4_g(alpha): return ((1 - alpha)**(-3) - 1)/3

# Diffusion model
def dm1_g(alpha): return alpha**2                                               # Parabolic law (1D Diffusion) (DM1)
def dm2_g(alpha): return alpha + ((1 - alpha) * np.log(1 - alpha))              # Valensi (2D diffusion) (DM2)
def dm3_g(alpha): return (1-(1-alpha)**(1/3))**2                                # Jander (3D Diffusion)
def dm4_g(alpha): return (1 - 2*alpha/3) - (1 - alpha)**(2/3)                   # Ginstling - Broushtein (3D Diffusion) (DM3)
def dm5_g(alpha): return ((1-alpha)**(-1/3) - 1)**2                             # Zhuravlev-Lesokhin-Tempelman

# Nucleation and growth model
def ng15_g(alpha): return (-np.log(1 - alpha))**(2/3)                           # Avrami - Erofeev (n = 1.5) (NG1.5)
def ng2_g(alpha): return (-np.log(1 - alpha))**(1/2)                            # Avrami - Erofeev (n = 2) (NG2)
def ng3_g(alpha): return (-np.log(1 - alpha))**(1/3)                            # Avrami - Erofeev (n = 3) (NG3)
def ng4_g(alpha): return (-np.log(1 - alpha))**(1/4)                            # Avrami - Erofeev (n = 4) (NG4)
def b1_g(alpha): return np.log(alpha/(1-alpha))                                 # Prout-Tompkins
def p23_g(alpha): return alpha**(3/2)                                           # Power law
def p2_g(alpha): return alpha**(1/2)                                            # Power law
def p3_g(alpha): return alpha**(1/3)                                            # Power law
def p4_g(alpha): return alpha**(1/4)                                            # Power law
def e1_g(alpha): return np.log(alpha)                                           # Exponent Power first-order
def e2_g(alpha): return np.log(alpha**2)                                        # Exponent Power second-order

# Geometrical contraction models
def r2_g(alpha): return 1 - (1 - alpha)**(1/2)                                  # Contracting cylinder
def r3_g(alpha): return 1 - (1 - alpha)**(1/3)                                  # Contracting volume

f_funcs = {
    "CR0":  cr0_f, "CR05": cr05_f, "CR1": cr1_f, "CR15": cr15_f, "CR2": cr2_f,"CR25": cr25_f, "CR3": cr3_f, "CR35": cr35_f, "CR4": cr4_f,
    "DM1": dm1_f, "DM2": dm2_f, "DM3": dm3_f, "DM4": dm4_f, "DM5": dm5_f,
    "NG15": ng15_f, "NG2": ng2_f, "NG3": ng3_f, "NG4": ng4_f, "B1": b1_f, "P23":p23_f, "P2":p2_f,"P3":p3_f,"P4":p23_f,"E1":e1_f, "E2": e2_f,
    "R2": r2_f, "R3": r3_f
}

g_funcs = {
    "CR0":  cr0_g, "CR05": cr05_g, "CR1": cr1_g, "CR15": cr15_g, "CR2": cr2_g, "CR25": cr25_g, "CR3": cr3_g, "CR35": cr35_g, "CR4": cr4_g,
    "DM1": dm1_g, "DM2": dm2_g, "DM3": dm3_g, "DM4": dm4_g, "DM5": dm5_g,
    "NG15": ng15_g, "NG2": ng2_g, "NG3": ng3_g, "NG4": ng4_g, "B1": b1_g, "P23":p23_g, "P2":p2_g,"P3":p3_g,"P4":p23_g,"E1":e1_g, "E2": e2_g,
    "R2": r2_g, "R3": r3_g
}

# Estas son las que se usan para CR. Se han quitado E1, E2 (que son casi las mismas) y tambien se ha quitado B1 pq en su forma integral es 0 todo el rato...
g_funcss = {
    "CR0":  cr0_g, "CR05": cr05_g, "CR1": cr1_g, "CR15": cr15_g, "CR2": cr2_g, "CR25": cr25_g, "CR3": cr3_g, "CR35": cr35_g, "CR4": cr4_g,
    "DM1": dm1_g, "DM2": dm2_g, "DM3": dm3_g, "DM4": dm4_g, "DM5": dm5_g,
    "NG15": ng15_g, "NG2": ng2_g, "NG3": ng3_g,"NG4": ng4_g, "P23":p23_g, "P2":p2_g,"P3":p3_g,"P4":p23_g,
    "R2": r2_g, "R3": r3_g
}

## FUNCIONES AUXILIARES

def cargar_csv_trios(ruta_archivo):
    """
    Se cargan los datos del archivo .csv a un dataframe (df) y se sustituyen las ',' que hay en los valores de las columnas por '.'
    Ademas se convierten dichos valores a float y se devuelve el df con las modificaciones.
    """
    df = pd.read_csv(ruta_archivo, sep=';')
    for col in ['Temperature T(c)', 'Time t (min)', 'Weight (mg)', 'Weight (%)', 'Heat Flow Q (mW)', 'Heat Flow (Normalized) Q (W/g)']:
        df[col] = df[col].str.replace(',', '.').astype(float)
    return df

def procesar_datos_trios(df):
    """
    Procesa el DataFrame de datos termogravimétricos para extraer y calcular las variables necesarias.

    Realiza las siguientes operaciones:
    1. Convierte las columnas relevantes del DataFrame a arrays de NumPy
    2. Transforma la temperatura de Celsius a Kelvin
    3. Calcula la fracción de conversión (alpha)
    4. Filtra los valores de alpha para mantener solo los válidos (0 < alpha < 1)

    Args:
        df (pd.DataFrame): DataFrame cargado con los datos del experimento, típicamente obtenido
                          de la función cargar_csv_trios(). Debe contener las columnas:
                          - 'Time t (min)'
                          - 'Temperature T(c)'
                          - 'Weight (mg)'
                          - 'Weight (%)'
                          - 'Heat Flow Q (mW)'
                          - 'Heat Flow (Normalized) Q (W/g)'

    Returns:
        dict: Diccionario con los siguientes arrays procesados:
              - 'time': Tiempo en minutos (array completo)
              - 'temperature_k': Temperatura en Kelvin (filtrado por alpha válido)
              - 'alpha': Fracción de conversión (filtrado, solo valores 0 < alpha < 1)
              - 'weight_mg': Peso en miligramos (array completo)
              - 'weight_percent': Porcentaje de peso (array completo)
              - 'heat_flow_q': Flujo de calor en mW (array completo)
              - 'heat_flow_normalized': Flujo de calor normalizado (W/g) (array completo)
              - 'temperature': Temperatura en Celsius (array completo)

    Notas:
        - La fracción de conversión (alpha) se calcula como:
          alpha = (peso_inicial - peso_actual) / (peso_inicial - peso_final) o alpha = (100 - peso_actual_porcentaje) / 100
        - Los valores de alpha se recortan al rango [0, 1] usando np.clip
        - Solo se devuelven los puntos donde 0 < alpha < 1 para temperatura_k y alpha
    """
    # Convertir las columnas de Pandas a arrays de Numpy para mejor rendimiento
    time = np.array(df['Time t (min)'])
    temperature = np.array(df['Temperature T(c)'])
    weight_percent = np.array(df['Weight (%)'])
    weight_mg = np.array(df['Weight (mg)'])
    heat_flow_q = np.array(df['Heat Flow Q (mW)'])
    heat_flow_normalized = np.array(df['Heat Flow (Normalized) Q (W/g)'])

    # Conversión de temperatura a Kelvin (necesario para cálculos termodinámicos)
    temperature_k = temperature + 273

    # Calcular fracción de conversión (alpha):
    alpha = (100 - weight_percent)/100

    # Filtrado de valores de alpha validos
    valid_alpha_mask = (alpha > 0) & (alpha < 1)

    # Devuelve un diccionario estructurado con los datos procesados
    return {
        'time': time[valid_alpha_mask],
        'temperature_k': temperature_k[valid_alpha_mask],
        'alpha': alpha[valid_alpha_mask],
        'weight_mg': weight_mg [valid_alpha_mask],
        'weight_percent': weight_percent[valid_alpha_mask],
        'heat_flow_q': heat_flow_q [valid_alpha_mask],
        'heat_flow_normalized': heat_flow_normalized [valid_alpha_mask],
        'temperature': temperature [valid_alpha_mask]
    }


def calcular_dtg(temperature, weight_mg):
    """
    Calcula la curva de Termogravimetría Derivada (DTG) usando el método numérico de diferencias finitas.

    La DTG representa la tasa de cambio del peso en función de la temperatura (dm/dT), que es útil
    para identificar puntos de inflexión y temperaturas características en el análisis térmico.

    Args:
        temperature (np.array): Array de temperaturas en °C o K (debe ser monótonamente creciente)
        weight_mg (np.array): Array de pesos en miligramos correspondiente a cada temperatura

    Returns:
        np.array: Curva DTG (derivada dm/dT) con las siguientes características:
                 - Mismo tamaño que los arrays de entrada
                 - Valores NaN e infinitos reemplazados por 0
                 - Suavizado en los extremos mediante diferencias unilaterales

    Notas:
        - Método numérico: Diferencias centrales para puntos interiores, diferencias hacia adelante/atrás para extremos
        - Manejo de casos especiales:
          * División por cero: Se devuelve 0 en ese punto
          * Valores NaN/Inf: Se convierten a 0
        - La precisión depende del espaciado uniforme entre puntos de temperatura
    """
    # Inicializar array de resultados con ceros
    dtg = np.zeros_like(weight_mg)

    # Cálculo para puntos interiores
    for i in range(1, len(temperature) - 1):
        delta_temp = temperature[i] - temperature[i-1]
        dtg[i] = (weight_mg[i] - weight_mg[i-1]) / delta_temp if delta_temp != 0 else 0

    # Tratamiento especial para extremos:
    delta_temp_start = temperature[1] - temperature[0]
    dtg[0] = (weight_mg[1] - weight_mg[0]) / delta_temp_start if delta_temp_start != 0 else 0

    delta_temp_end = temperature[-1] - temperature[-2]
    dtg[-1] = (weight_mg[-1] - weight_mg[-2]) / delta_temp_end if delta_temp_end != 0 else 0

    # Limpieza de valores numéricamente problemáticos
    return np.nan_to_num(dtg, nan=0.0, posinf=0.0, neginf=0.0)

def suavizar_dtg(dtg, window_size):
    """
   Aplica un filtro de promedio móvil para suavizar datos de Termogravimetría Derivada (DTG).

   Args:
       dtg (np.array): Array con valores de DTG (derivada de peso vs temperatura)
       window_size (int): Tamaño de la ventana de suavizado (debe ser impar y >1)

   Returns:
       np.array: Array suavizado con mismo tamaño que la entrada

   Notas:
       - Implementación mediante convolución con kernel uniforme (cada punto
         dentro de la ventana tiene igual peso en el cálculo del promedio)
       - Modo 'same' mantiene dimensiones originales
       - Ventanas pares pueden causar desplazamiento de picos
   """
    return np.convolve(dtg, np.ones(window_size) / window_size, mode='same')


def dividir_en_subgrupos(temperature, temperature_k, alpha, weight_mg, time, heat_flow_q, temp_subgrupos):
    """
    Divide los datos termogravimétricos en subgrupos según temperaturas de referencia.

    Args:
        temperature (np.array): Array de temperaturas en °C
        temperature_k (np.array): Array de temperaturas en Kelvin
        alpha (np.array): Array de fracciones de conversión
        weight_mg (np.array): Array de pesos en miligramos
        time (np.array): Array de tiempos en minutos
        heat_flow_q (np.array): Array de flujo calorífico en mW
        temp_subgrupos (list): Lista de temperaturas para dividir los datos

    Returns:
        tuple: (subgrupos, indices) donde:
            - subgrupos (dict): Diccionario con arrays divididos para cada variable
            - indices (list): Posiciones donde se realizaron las divisiones

    Notas:
        - Usa encontrar_mas_cercano() para localizar puntos de división
        - Elimina duplicados y ordena los índices automáticamente
        - Estructura de subgrupos: {'temp': [subarr1, subarr2...], ...}
    """
    # Encontrar los índices de las temperaturas seleccionadas o las más cercanas
    indices = []
    for temp in temp_subgrupos:
        idx = (np.abs(temperature - temp)).argmin()

        temp_cercana = temperature[idx]
        # print( f"La temperatura más cercana a {temp:.2f} en el vector de Temperatura es {temp_cercana:.2f} en el índice {idx}")
        indices.append(idx)  # Añade el indice idx (el indice de la temp mas cercana) a lista de indices

    # Ordenar los índices por si no están en orden y eliminar duplicados
    indices = sorted(set(indices))

    # Dividir los arrays en subgrupos según esos índices. Retorna un diccionario que contiene los valores de los subgrupos de
    # cada una de las variables. subgrupos['temp'] o subrupos.get('temp') = [[valores subgrupo 1], [valores subgrupo 2]...]
    subgrupos = {
        'temp': np.split(temperature, indices),
        'temp_K': np.split(temperature_k, indices),
        'alpha': np.split(alpha, indices),
        'weight': np.split(weight_mg, indices),
        'time': np.split(time, indices),
        'heat_flow_q': np.split(heat_flow_q, indices)
    }
    return subgrupos, indices


def graficar_multieje(x_data, y_data, ruta_archivo, particiones=None):
    """
    Genera y guarda un gráfico multivariable con ejes Y múltiples a partir de datos termogravimétricos.

    Args:
        eje_x (np.array): Array de temperaturas en °C o tiempo en min.
        y_data (dict): Diccionario con:
                       - keys: nombres de las series (ej: 'Weight (%)')
                       - values: arrays de datos para ejes Y
        ruta_archivo (str): Ruta del archivo original (para nombrar el gráfico)
        particiones (list, opcional): Índices para líneas verticales de división
    Returns:
        None: Guarda el gráfico como archivo .png y muestra la ruta de guardado

    Notas:
        - Asigna automáticamente colores distintos a cada eje Y
        - Crea hasta 8 ejes Y desplazados a la derecha
        - Las líneas de partición son grises discontinuas
        - Guarda en carpetas diferentes para Componentes/Biomasa
        - Resolución: 300 DPI (alta calidad)
    """

    # Configuración del gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colores = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    ejes_y = [ax1]

    # Eje X común (temperatura)
    ax1.set_xlabel('Temperature T(°C)', fontsize = 12)

    # Generar ejes Y
    for i, (label, y_values) in enumerate(y_data.items()):
        if i > 0:
            ax = ax1.twinx()
            ax.spines["right"].set_position(("axes", 1 + 0.12 * (i - 1)))
            ejes_y.append(ax)
        else:
            ax = ax1

        ax.plot(x_data, y_values, color=colores[i % 8], label='_nolegend_')
        ax.set_ylabel(label, color=colores[i % 8], fontsize=12)
        ax.tick_params(axis='y', labelcolor=colores[i % 8], labelsize=8)
    
    # Añadir particiones
    if particiones:
        for idx in particiones:
            valor_x = x_data[idx]
            ax1.axvline(x=valor_x, color='lightgray', linestyle='--', alpha=0.7,
                        label=f'Partición en {valor_x:.1f}°C')
            ax1.legend(loc='best')

    # Configuración general
    nombre = os.path.splitext(os.path.basename(ruta_archivo))[0]
    plt.title(f"Gráfico de {nombre}")
    fig.tight_layout()

    # Guardado automático
    tipo = "Componentes" if "Componentes" in ruta_archivo else "Muestras"
    ruta_guardado = f"Resultados/F0_Graficas{tipo}/{os.path.basename(os.path.dirname(ruta_archivo))}/{nombre}.png"

    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gráfico guardado en: {ruta_guardado}")


def calcular_energia(heat_flow_q, time, ruta_archivo, t_corte_min = 11.88):
    """
    Integra el flujo de calor normalizado y guarda las energías (kJ/kg) antes y después
    de un punto de corte (~200 °C) en un Excel dentro de:
    Resultados/CurvaDSC/<Componentes|Biomasa>/<VelocidadCalentamientoX>/<nombre>.xlsx

    Args:
        heat_flow_q (np.array): Flujo de calor normalizado en W/g (J·s⁻¹·g⁻¹).
        time (np.array): Tiempo en minutos.
        ruta_archivo (str): Ruta del archivo de origen para inferir carpeta y nombre.
        t_corte_min (float): Punto de corte en minutos (por defecto 11.88 ≈ 200 °C).

    Returns:
        None. Crea el Excel y muestra la ruta guardada.
    """
    # Conversión de unidades
    heat_flow_w = heat_flow_q * 1000  # (W/g) -> (W/kg)
    time_s = time * 60  # min -> s
    t_corte_s = t_corte_min * 60

    # Máscaras
    mask_antes = time_s <= t_corte_s
    mask_despues = time_s >= t_corte_s

    # Integración (regla del trapecio)
    energia_antes = np.trapezoid(heat_flow_w[mask_antes], time_s[mask_antes]) / 1000  # kJ/kg
    energia_despues = np.trapezoid(heat_flow_w[mask_despues], time_s[mask_despues]) / 1000  # kJ/kg
    energia_total = np.trapezoid(heat_flow_w, time_s) / 1000  # kJ/kg

    # DataFrame de salida
    df_out = pd.DataFrame({
        "Tramo": ["Antes de 200 °C", "Después de 200 °C", "Total"],
        "t_inicio_min": [float(time[0]), float(t_corte_min), float(time[0])],
        "t_fin_min": [float(t_corte_min), float(time[-1]), float(time[-1])],
        "Energía (kJ/kg)": [energia_antes, energia_despues, energia_total],
    })

    # Construcción de ruta de guardado
    nombre = os.path.splitext(os.path.basename(ruta_archivo))[0]
    subcarpeta = os.path.basename(os.path.dirname(ruta_archivo))  # 'VelocidadCalentamientoX'

    # Detectar tipo: Componentes o Biomasa
    ruta_lower = ruta_archivo.lower()
    if "componentes" in ruta_lower:
        tipo = "Componentes"
    elif "biomasa" in ruta_lower:
        tipo = "Biomasa"
    else:
        # Fallback: si no se detecta, se guarda en Biomasa por defecto
        tipo = "Biomasa"

    dir_salida = os.path.join("Resultados", "F0_CurvaDSC", tipo, subcarpeta)
    os.makedirs(dir_salida, exist_ok=True)
    ruta_salida = os.path.join(dir_salida, f"{nombre}.xlsx")

    # Guardar
    df_out.to_excel(ruta_salida, index=False)
    print(f"Energías DSC guardadas en: '{ruta_salida}'")




# ========================= FLUJO GENERAL ====================================
# FASE 0: TRATAMIENTO DE DATOS EXPERIMENTALES
def tratamiento_datos_exp (directorios_base):
    """
    Procesa en lote todos los .csv de los directorios base, genera variables derivadas,
    actualiza el diccionario global `DATOS_PROCESADOS`, guarda gráficos multieje y
    exporta a Excel la energía DSC integrada por tramos.

    Args:
        directorios_base (list): Lista de carpetas raíz a recorrer recursivamente.
            Estructura esperada (ejemplo):
                <DatosComponentes|DatosBiomasa|DatosMuestras>/
                    VelocidadCalentamientoX/
                        <nombre>.csv
            donde X es la velocidad de calentamiento (°C/min) usada para calcular dα/dt.

    Returns:
        None: Actualiza `DATOS_PROCESADOS` y crea ficheros en la carpeta Resultados/*.

    Notas:
        - `beta` se infiere de la subcarpeta 'VelocidadCalentamientoX'.
        - Suavizado: ventana 400 para DTG (mg) y 200 para dα/dt.
        - `temp_seleccionadas` define divisiones (particiones) por nombre de archivo para el graficado
          y para segmentar datos en `dividir_en_subgrupos`.
        - Requiere las funciones auxiliares:
          `cargar_csv_trios`, `procesar_datos_trios`, `calcular_dtg`, `suavizar_dtg`,
          `dividir_en_subgrupos`, `graficar_multieje`, `calcular_energia`.
        - Salidas:
            * PNG de gráficos en Resultados/Graficas<Componentes|Muestras>/
            * Excel de energía DSC en Resultados/CurvaDSC/<Componentes|Biomasa>/VelocidadCalentamientoX/
    """
    temp_seleccionadas = {
        'celulosa_5_aire': [290, 335, 530],
        'celulosa_5_n2': [295, 340, 500],
        'lignina_5_aire': [200, 420],
        'lignina_5_n2': [190, 600],
        'xilano_5_aire': [240, 265, 450],
        'xilano_5_n2': [245, 280, 600],
        'celulosa_15_aire': [310, 350, 575],
        'celulosa_15_n2': [315, 360, 600],
        'lignina_15_aire': [225, 630],
        'lignina_15_n2': [200, 580],
        'xilano_15_aire': [255, 285, 520],
        'xilano_15_n2': [270, 290, 600],
        'celulosa_30_aire': [320, 375, 600],
        'celulosa_30_n2': [330, 380, 600],
        'lignina_30_aire': [225, 850],
        'lignina_30_n2': [200, 440],
        'xilano_30_aire': [265, 300, 645],
        'xilano_30_n2': [280, 310, 600],
        'muestra1': [300, 350, 600],
        'muestra2': [270, 300, 350, 500],
        'muestra3': [200, 270, 600],
        'muestra4': [280, 300, 550],
        'muestra5': [250, 350, 520],
    }

    for directorio in directorios_base:
        for raiz, _, archivos in os.walk(directorio):
            # Recorrer los archivos en la lista de archivos de esa carpeta
            for archivo in archivos:
                # Verificar si el archivo es un archivo .csv
                if archivo.endswith('.csv'):
                    ruta_completa = os.path.join(raiz, archivo)

                    # Obtener componentes de la ruta
                    partes = raiz.split(os.sep)
                    tipo_dato = partes[0]  # 'DatosComponentes' o 'DatosMuestras'
                    subcarpeta = partes[1]  # 'VelocidadCalentamientoX'
                    nombre = os.path.splitext(archivo)[0]  # 'celulosa_15_aire'
                    beta = int(subcarpeta.replace("VelocidadCalentamiento", ""))

                    # Cargar y procesar datos
                    df = cargar_csv_trios(ruta_completa)
                    datos_procesados = procesar_datos_trios(df)

                    # Calcular DTG suavizado
                    dtg = calcular_dtg(datos_procesados['temperature'], datos_procesados['weight_mg'])
                    dtg_suavizado = suavizar_dtg(dtg, 400)

                    dalpha_dT = calcular_dtg(datos_procesados['temperature'], datos_procesados['alpha'])
                    dalpha_dt = suavizar_dtg(dalpha_dT * beta,200)

                    # Obtener temperaturas seleccionadas
                    temps = temp_seleccionadas.get(nombre, [])

                    # Almacenar en estructura global
                    if subcarpeta not in DATOS_PROCESADOS[tipo_dato]:
                        DATOS_PROCESADOS[tipo_dato][subcarpeta] = {}

                    DATOS_PROCESADOS[tipo_dato][subcarpeta][nombre] = {
                        'time': datos_procesados['time'],
                        'temperature_k': datos_procesados['temperature_k'],
                        'alpha': datos_procesados['alpha'],
                        'weight_mg': datos_procesados['weight_mg'],
                        'weight_percent': datos_procesados['weight_percent'],
                        'heat_flow_q': datos_procesados['heat_flow_q'],
                        'heat_flow_normalized': datos_procesados ['heat_flow_normalized'],
                        'temperature': datos_procesados['temperature'],
                        'DTG_suavizado': dtg_suavizado,
                        'dalpha_dt': dalpha_dt,
                        'temp_seleccionadas': temps,
                        'dalpha_dT': dalpha_dT,
                        'dataframe': df  # Opcional
                    }


                    """# Encontrar temperatura del pico máximo de DTG - Se ha usado para aplicar la función de Kissinger de A
                    idx_pico_max = np.argmax(dalpha_dt)
                    temp_pico_max = datos_procesados['temperature'][idx_pico_max]

                    print(
                        f"[{nombre}] Máximo DTG: {dalpha_dt[idx_pico_max]:.4f} mg/min "
                        f"@ {temp_pico_max:.1f} °C (β={beta}°C/min) \n"
                    )"""


                    subgrupos, indices = dividir_en_subgrupos(datos_procesados['temperature'], datos_procesados['temperature_k'], datos_procesados['alpha'],
                                                              datos_procesados['weight_mg'], datos_procesados['time'], datos_procesados['heat_flow_q'], temps)

                    # Preparar datos para el gráfico
                    columnas_y = {
                        #'Weight (%)': datos_procesados['weight_percent'],
                        'α': datos_procesados['alpha'],
                        'DTG': dalpha_dt,
                        'Heat Flow (W/g)': datos_procesados['heat_flow_normalized'],
                    }
                    graficar_multieje(x_data=datos_procesados['temperature'], y_data = columnas_y, ruta_archivo= ruta_completa, particiones=indices)

                    # Obviar los resultados de energia de los componentes.
                    calcular_energia(datos_procesados['heat_flow_normalized'], datos_procesados['time'], ruta_completa)


# FASE 1.0: APLICAR FWO, STARINK Y FRIEDMAN
def aplicar_fwo_starink_friedman(componente, atmosfera, alphas=np.arange(0.1, 0.901, 0.05)):
    """
    Aplica los métodos isoconversionales FWO y Starink (más el diferencial de Friedman)
    para un componente y atmósfera determinados, y guarda resultados y gráficos.

    Args:
        componente (str): 'celulosa', 'xilano' o 'lignina'.
        atmosfera (str): 'aire' o 'n2'.
        alphas (array-like): Fracciones de conversión a evaluar (recomendado 0.05–0.95).

    Returns:
        pandas.DataFrame: Tabla con Ea (kJ/mol), R² y datos de ajuste por α.

    Efectos:
        - Requiere datos previos en `DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamientoX'][<clave>]`.
        - Guarda gráficos y Excel en:
          `Resultados/FWOandStarink/<Atmosfera>/<Componente>/FWO_Starink_<componente>_<atmosfera>.(png|xlsx)`

    Notas:
        - FWO:    ln(β)        = C - (1.052 * Ea / R) * (1/T)
        - Starink:ln(β/T^1.92) = C - (1.0008 * Ea / R) * (1/T)
        - Friedman: ln(dα/dt)  = C - Ea/(R*T)
        - Se requieren ≥3 velocidades de calentamiento para cada α.
        - Usa la constante universal de los gases `R` (debe existir en el entorno).
    """
    # Velocidades disponibles esperadas en DATOS_PROCESADOS
    betas_disponibles = [5, 15, 30]

    resultados = {
        'alpha': [],
        'time': [],
        'Ea_FWO_kJ_mol': [],
        'R2_FWO': [],
        'Ea_Starink_kJ_mol': [],
        'R2_Starink': [],
        'Ea_Friedman_kJ_mol': [],
        'R2_Friedman': [],
        'ln_beta_FWO': [],
        'ln_beta_T192_Starink': [],
        'ln_Friedman': [],
        'inv_temp': [],
        'intercepto_starink': [],
        'intercepto_fwo': []
    }

    # Recopilar datos del componente a diferentes β
    datos_componente = {}
    for beta in betas_disponibles:
        clave = f"{componente}_{beta}_{atmosfera}".lower()
        velocidad_key = f"VelocidadCalentamiento{beta}"

        # Verificar si existen datos para esta combinación
        try:
            datos = DATOS_PROCESADOS['DatosComponentes'][velocidad_key][clave]
            datos_componente[beta] = {
                'temperature_k': datos['temperature_k'],
                'alpha': datos['alpha'],
                'time': datos['time'],
                'dalpha_dt': datos['dalpha_dt'],
            }
        except KeyError:
            print(f"Advertencia: Datos no encontrados para {clave}")
            continue

    if len(datos_componente) < 3:
        raise ValueError(f"Se requieren 3 velocidades para {componente} en {atmosfera}")       # Con 2 velocidades, el ajuste siempre es perfecto. No valido.

    # Cálculos por cada α
    for alpha in alphas:
        temps = []
        times = []
        alfas = []
        ln_betas_fwo = []
        ln_betas_T192_starink = []
        ln_friedman = []

        # Para cada β, tomar punto más cercano al α pedido
        for beta, datos in datos_componente.items():
            try:
                idx = np.abs(datos['alpha'] - alpha).argmin()
                T = datos['temperature_k'][idx]
                t = datos['time'][idx]
                a = datos['alpha'][idx]
                dadt = datos ['dalpha_dt'][idx]

                if not np.isnan(T) and T > 0:
                    temps.append(T)
                    times.append(t)
                    alfas.append(a)
                    ln_betas_fwo.append(np.log(beta))
                    ln_betas_T192_starink.append(np.log(beta/T**1.92))
                    ln_friedman.append(np.log(dadt))

            except (IndexError, ValueError):
                continue

        # Requerir 3 puntos para el ajuste lineal
        if len(temps) >= 3:
            inv_temps = 1 / np.array (temps)

            # FWO: ln(β) vs 1/T  ->  pendiente = -(1.052*Ea/R)
            slope_fwo, intercept_fwo, r_fwo, _, _ = linregress(inv_temps, ln_betas_fwo)
            Ea_fwo = -slope_fwo * R / 1.052  # Energía de activación en J/mol

            # Starink: ln(β/T^1.92) vs 1/T -> pendiente = -(1.0008*Ea/R)
            slope_starink, intercept_starink, r_starink, _, _ = linregress(inv_temps, ln_betas_T192_starink)
            Ea_starink = -slope_starink * R / 1.0008

            # Friedman: ln(dα/dt) vs 1/T -> pendiente = -Ea/R
            slope_friedman, intercept_friedman, r_friedman, _, _ = linregress(inv_temps, ln_friedman)
            Ea_friedman = -slope_friedman * R

            resultados['alpha'].append(alpha)
            resultados['time'].append(np.array(times))
            resultados['Ea_FWO_kJ_mol'].append(Ea_fwo / 1000)
            resultados['R2_FWO'].append(r_fwo ** 2)
            resultados['Ea_Starink_kJ_mol'].append(Ea_starink / 1000)
            resultados['R2_Starink'].append(r_starink ** 2)
            resultados['Ea_Friedman_kJ_mol'].append(Ea_friedman / 1000)
            resultados['R2_Friedman'].append(r_friedman ** 2)
            resultados['ln_beta_FWO'].append(ln_betas_fwo)
            resultados['ln_beta_T192_Starink'].append(ln_betas_T192_starink)
            resultados['ln_Friedman'].append (np.array(ln_friedman))
            resultados['inv_temp'].append(inv_temps)
            resultados['intercepto_starink'].append(intercept_starink)  # para calculo del factor preexponencial A por Starink
            resultados['intercepto_fwo'].append(intercept_fwo)      # para calculo del factor preexponencial A por FWO

    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)

    # Generación y guardado de gráficos
    directorio_base = f"Resultados/F1.0_FWOStarinkFriedman/{atmosfera.capitalize()}/{componente.capitalize()}"
    os.makedirs(directorio_base, exist_ok=True)

    # Generar gráficos
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 1))

    # Puntos y rectas por α (se usan los handles de ax1 para la leyenda)
    handles = []

    # Gráfico 1 , 2 y 3: FWO, KAS y Friedmann
    for _, row in df_resultados.iterrows():
        label = f"α={row['alpha']:.2f}"

        h1 = ax1.scatter(row['inv_temp'], row['ln_beta_FWO'], label=label)
        ax1.plot(row['inv_temp'], np.polyval(np.polyfit(row['inv_temp'], row['ln_beta_FWO'],1),row['inv_temp']),'--')

        ax2.scatter(row['inv_temp'], row['ln_beta_T192_Starink'], label=label)
        ax2.plot(row['inv_temp'], np.polyval(np.polyfit(row['inv_temp'], row['ln_beta_T192_Starink'],1), row['inv_temp']), '--')

        ax3.scatter(row['inv_temp'], row['ln_Friedman'], label=label)
        ax3.plot(row['inv_temp'], np.polyval(np.polyfit(row['inv_temp'], row['ln_Friedman'], 1), row['inv_temp']), '--')

        # Guardar un único handle por cada α (usamos solo los de ax1)
        handles.append(h1)

    # Etiquetas / títulos
    ax1.set_xlabel('1/T (K⁻¹)', fontsize = 16)
    ax1.set_ylabel('ln(β)', fontsize = 16)
    ax1.set_title(f'FWO: {componente.capitalize()} en {atmosfera.upper()}\nln(β) vs 1/T', fontsize = 18)

    ax2.set_xlabel('1/T (K⁻¹)', fontsize = 16)
    ax2.set_ylabel('ln(β/T¹·⁹²)', fontsize = 16)
    ax2.set_title(f'Starink: {componente.capitalize()} en {atmosfera.upper()}\nln(β/T¹·⁹²) vs 1/T', fontsize = 18)

    ax3.set_xlabel('1/T (K⁻¹)', fontsize = 16)
    ax3.set_ylabel('ln(d(α)/dt)', fontsize = 16)
    ax3.set_title(f'Friedman: {componente.capitalize()} en {atmosfera.upper()}\nln(d(α)/dt) vs 1/T', fontsize = 18)

    # Leyenda común externa (a la derecha de la figura)
    fig.legend(handles,
               [f"α={row['alpha']:.2f}" for _, row in df_resultados.iterrows()],
               loc='center right', title="Valores de α",
               bbox_to_anchor=(1.02, 0.75), title_fontsize=12, prop={'size':12})

    # Gráfico 4:  Comparativa Ea(α)
    ax4.plot(df_resultados['alpha'], df_resultados['Ea_FWO_kJ_mol'], 'o-', label='FWO')
    ax4.plot(df_resultados['alpha'], df_resultados['Ea_Starink_kJ_mol'], 'o-', label='Starink')
    ax4.plot(df_resultados['alpha'], df_resultados['Ea_Friedman_kJ_mol'], 'o-', label='Friedman')
    ax4.set_xlabel('α', fontsize = 16)
    ax4.set_ylabel('Ea (kJ/mol)', fontsize = 16)
    ax4.set_title(f'Energía de Activación vs Conversión\n{componente.capitalize()} en {atmosfera.upper()}', fontsize = 18)
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 0.97, 1])

    # Guardar gráficos
    ruta_grafico = f"{directorio_base}/FWO_Starink_Friedman_{componente}_{atmosfera}.png"
    fig.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Guardar resultados en Excel
    ruta_excel = f"{directorio_base}/FWO_Starink_Friedman_{componente}_{atmosfera}.xlsx"
    df_resultados.to_excel(ruta_excel, index=False)

    print(f"Resultados guardados en:\n- Gráfico: {ruta_grafico}\n- Datos: {ruta_excel}")

    return df_resultados

def fase1_isoconversional():
    """
    Lanza el análisis isoconversional (FWO + Starink + Friedman) para
    todos los componentes y atmósferas soportados y guarda resultados.

    Recorre:
        componentes = ['celulosa', 'xilano', 'lignina']
        atmosferas  = ['aire', 'n2']

    Efectos:
        - Almacena el DataFrame resultante por combinación en
          `RESULTADOS_GLOBALES['FWO_Starink_Friedman'][<componente>][<atmósfera>]`.
        - Los gráficos y Excel se guardan automáticamente por `aplicar_fwo_starink_friedman`.
    """
    componentes = ['celulosa', 'xilano', 'lignina']
    atmosferas = ['aire', 'n2']

    for componente in componentes:
        # Inicializar estructura para el componente
        if componente not in RESULTADOS_GLOBALES["FWO_Starink_Friedman"]:
            RESULTADOS_GLOBALES["FWO_Starink_Friedman"][componente] = {}

        for atmosfera in atmosferas:
            try:
                print(f"\nProcesando {componente} en {atmosfera}...")

                # Ejecutar análisis (función existente)
                resultados = aplicar_fwo_starink_friedman(componente, atmosfera)

                # Almacenar en estructura global (mismo formato que DATOS_PROCESADOS)
                RESULTADOS_GLOBALES["FWO_Starink_Friedman"][componente][atmosfera] = {
                    'resultados': resultados
                }

            except Exception as e:
                print(f"Error en {componente}-{atmosfera}: {str(e)}")


# FASE 1.1: APLICAR CRIADO
def calcular_z_alpha_criado_exp(df_fwo_starink, beta, metodo='FWO'):
    """
    Calcula Z(α) experimental de Criado usando las Ea obtenidas por FWO o Starink
    (y las series diferenciales de Friedman) para una velocidad de calentamiento β dada.

    Args:
        df_fwo_starink (pandas.DataFrame): Resultados del análisis isoconversional
            (salida de `analizar_fwo_starink_friedman`), con columnas:
                - 'alpha'
                - 'Ea_FWO_kJ_mol' y/o 'Ea_Starink_kJ_mol'
                - 'inv_temp' (arrays por α)
                - 'ln_Friedman' (arrays por α)
        beta (float): Velocidad de calentamiento [°C/min]. Soportado: 5, 15, 30.
        metodo (str): 'FWO' o 'Starink' para seleccionar la columna de Ea.

    Returns:
        tuple:
            - alpha_values (np.ndarray): Valores de α usados.
            - Z_diff (np.ndarray): Forma diferencial normalizada f(α)/f(0.5).
            - Z_diff_int (np.ndarray): Forma integral*diferencial Z(α)=(dα/dt)·π(x)·T/β.
            - Z_int (np.ndarray): Forma integral normalizada g(α)/g(0.5).

    Notas:
        - x = Ea/(R·T). π(x) se aproxima mediante Senum–Yang.
        - Se usa ln(dα/dt) de Friedman para construir las formas experimentales.
        - Requiere la constante de los gases R en el entorno.
    """
    # 1) Extraer α y Ea (J/mol) del DataFrame isoconversional
    alpha_values = df_fwo_starink['alpha'].values
    Ea_values = df_fwo_starink[f'Ea_{metodo}_kJ_mol'].values * 1000  # Convertir a J/mol

    # 2) Selección de índice por β (orden esperado 5,15,30)
    # Dado que Temperatura y ln_dadt dentro de df_fwo_starink son dataframes de 3 valores, se accedera a cada uno de ellos dependiendo del valor de beta
    beta2idx = {5:0, 15:1, 30:2}
    col = beta2idx[beta]

    # 3) Recuperar T y dα/dt correspondientes a esta β
    inv_T_values = df_fwo_starink['inv_temp'].str[col]
    T_values = np.array(1.0/inv_T_values.values)

    ln_dadt_values = df_fwo_starink['ln_Friedman'].str[col]
    dadt_values = np.array (np.exp (ln_dadt_values.values))

    # 4) Cálculos comunes
    x = Ea_values/ (R * T_values)

    # Aproximación de Senum–Yang para π(x)
    pi_x = (x ** 3 + 18 * x ** 2 + 88 * x + 96) / (x ** 4 + 20 * x ** 3 + 120 * x ** 2 + 240 * x + 120)

    # --- Forma integral*diferencial (Z_diff_int) ---
    Z_diff_int = (dadt_values * pi_x * T_values) / beta

    # --- Forma diferencial normalizada (Z_diff = f(α)/f(0.5)) ---
    idx_alpha_05 = np.abs(alpha_values - 0.5).argmin()
    dadt_05 = dadt_values[idx_alpha_05]
    x_05 = Ea_values[idx_alpha_05] / (R * T_values[idx_alpha_05])
    Z_diff = (dadt_values / dadt_05) * (np.exp(x) / np.exp(x_05))

    # --- Forma integral normalizada (Z_int = g(α)/g(0.5)) ---
    # π(x_05)
    pi_x05 = (x_05 ** 3 + 18 * x_05 ** 2 + 88 * x_05 + 96) / (x_05 ** 4 + 20 * x_05 ** 3 + 120 * x_05 ** 2 + 240 * x_05 + 120)  # Se podría arreglar creando una funcion pi(x)

    px = (np.exp(-x) / x) * pi_x
    px_05 = (np.exp(-x_05) / x_05) * pi_x05
    Z_int = (px * Ea_values) / (px_05 * Ea_values[idx_alpha_05])

    # 5) Limpieza numérica
    # Manejar posibles divisiones por cero o valores inválidos
    Z_diff = np.nan_to_num(Z_diff, nan=0.0, posinf=0.0, neginf=0.0)
    Z_diff_int = np.nan_to_num(Z_diff_int, nan=0.0, posinf=0.0, neginf=0.0)
    Z_int = np.nan_to_num(Z_int, nan=0.0, posinf=0.0, neginf=0.0)

    return alpha_values, Z_diff, Z_diff_int, Z_int

def calcular_master_plots_teoricos(alphas=np.linspace(0.1, 0.901, 100)):
    """
    Genera los master plots teóricos (formas diferencial, integral e integral*diferencial)
    para todos los modelos cinéticos disponibles en los diccionarios `f_funcs` y `g_funcs`.

    Args:
        alphas (np.ndarray): Valores de conversión α a evaluar (recomendado 0.1–0.9).

    Returns:
        tuple:
            - teorico_diff (dict): f(α)/f(0.5) por modelo.
            - teorico_diff_int (dict): f(α)·g(α) por modelo.
            - teorico_int (dict): g(α)/g(0.5) por modelo.
            - alphas (np.ndarray): Los α usados.

    Notas:
        - Solo se calculan modelos presentes en ambos diccionarios `f_funcs` y `g_funcs`.
        - Si f(0.5) o g(0.5) = 0, se devuelve un vector de ceros para evitar divisiones por 0.
    """
    teorico_diff = {}
    teorico_diff_int = {}
    teorico_int = {}

    # Obtener modelos comunes en ambos diccionarios
    modelos_comunes = set(f_funcs.keys()) & set(g_funcs.keys())

    for model in modelos_comunes:
        # Obtener funciones f y g de los nuevos diccionarios
        f_func = f_funcs.get(model, None)
        g_func = g_funcs.get(model, None)

        if g_func is None or f_func is None:
            print(f"Aviso: faltan funciones para el modelo {model}")
            continue
        try:
            # Calculo de forma diferencial normalizada: f(α)/f(0.5)
            f_alpha = np.array([f_func(a) for a in alphas])
            f_05 = f_func(0.5)
            teorico_diff[model] = f_alpha / f_05 if f_05 != 0 else np.zeros_like(alphas)

            # Calculo de forma integral normalizada: g(α)/g(0.5)
            g_alpha = np.array([g_func(a) for a in alphas])
            g_05 = g_func (0.5)
            teorico_int[model] = g_alpha / g_05 if g_05 != 0 else np.zeros_like(alphas)

            # Calculo de forma combinada: f(α)*g(α)
            g_alpha = np.array([g_func(a) for a in alphas])
            teorico_diff_int[model] = f_alpha * g_alpha

        except Exception as e:
            print(f"Error procesando modelo {model}: {str(e)}")
            continue

    return teorico_diff, teorico_diff_int, teorico_int, alphas

def aplicar_criado(componente, atmosfera):
    """
    Aplica el método de Criado para identificar mecanismos de reacción comparando
    los master plots experimentales (a partir de FWO/Starink/Friedman) con los
    master plots teóricos. Genera gráficos interactivos (Plotly) y guarda los resultados.

    Args:
        componente (str): 'celulosa', 'xilano' o 'lignina'.
        atmosfera (str): 'aire' o 'n2'.

    Efectos:
        - Lee resultados de `RESULTADOS_GLOBALES['FWO_Starink_Friedman'][componente][atmosfera]['resultados']`.
        - Para β ∈ {5,15,30}:
            * Calcula Z(α) experimental por FWO y Starink.
            * Dibuja y guarda gráficos interactivos HTML con dropdown por modelo.
            * Sube la figura a Chart Studio.
        - Guarda:
            Resultados/Criado/<Atmósfera>/<Componente>/Criado_<componente>_<beta>_<atmósfera>.html
            y un archivo *_url.txt con el enlace de Chart Studio.

    Notas:
        - Requiere `plotly.graph_objects as go`, `plotly.subplots.make_subplots as make_subplots`,
          y `chart_studio.plotly as py` configurado (credenciales).
        - Depende de `generar_master_plots_teoricos` y `calcular_z_alpha_criado_exp`.
    """
    # Recuperar DataFrame isoconversional previamente calculado
    # Se llama al dataframe fwo-starink a pesar de que también contiene datos de friedman.
    df_fwo_starink = RESULTADOS_GLOBALES["FWO_Starink_Friedman"][componente][atmosfera]["resultados"]

    betas = [5, 15, 30]

    for beta in betas:
        try:
            # Obtener datos experimentales para esta β
            clave = f"{componente}_{beta}_{atmosfera}".lower()

            # Calcular Z(α) para ambos tipos
            alpha_fwo, Z_diff_fwo, Z_diff_int_fwo, Z_int_fwo = calcular_z_alpha_criado_exp(df_fwo_starink, beta, metodo='FWO')
            mask_fwo = (alpha_fwo >= 0.1) & (alpha_fwo <= 0.901)

            alpha_starink, Z_diff_starink, Z_diff_int_starink, Z_int_starink = calcular_z_alpha_criado_exp(df_fwo_starink, beta, metodo='Starink')
            mask_starink = (alpha_starink >= 0.1) & (alpha_starink <= 0.901)

        except KeyError:
            print(f"Advertencia: Datos no encontrados para {clave}")
            continue

        # Calcular curvas teóricas
        teorico_diff, teorico_diff_int, teorico_int, alphas_teo = calcular_master_plots_teoricos()

        # Crear figura con 3 subplots
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=('Forma Diferencial Normalizada','Formal Integral Normalizada', 'Forma Integral*Diferencial'),
                            horizontal_spacing=0.05)

        # Paleta por familias de modelos
        category_colors = {
            "CR": "#FFB3BA",  # Contracción
            "DM": "#BAFFC9",  # Difusión
            "NG": "#BAE1FF",  # Nucleación y crecimiento
            "B/P": "#FFFFBA", # Bifásico/Prouts
            "E/R": "#D0BAFF"  # Reacción de orden n (E) / Reacción (R)
        }

        # Atributos por modelo
        model_attributes = {
            # Modelos CR
            "CR0": {"symbol": "circle", "color": category_colors["CR"]},
            "CR05": {"symbol": "square", "color": category_colors["CR"]},
            "CR1": {"symbol": "diamond", "color": category_colors["CR"]},
            "CR15": {"symbol": "cross", "color": category_colors["CR"]},
            "CR2": {"symbol": "x", "color": category_colors["CR"]},
            "CR25": {"symbol": "triangle-up", "color": category_colors["CR"]},
            "CR3": {"symbol": "triangle-down", "color": category_colors["CR"]},
            "CR35": {"symbol": "pentagon", "color": category_colors["CR"]},
            "CR4": {"symbol": "hexagon", "color": category_colors["CR"]},

            # Modelos DM
            "DM1": {"symbol": "circle", "color": category_colors["DM"]},
            "DM2": {"symbol": "square", "color": category_colors["DM"]},
            "DM3": {"symbol": "diamond", "color": category_colors["DM"]},
            "DM4": {"symbol": "cross", "color": category_colors["DM"]},
            "DM5": {"symbol": "x", "color": category_colors["DM"]},

            # Modelos NG
            "NG15": {"symbol": "circle", "color": category_colors["NG"]},
            "NG2": {"symbol": "square", "color": category_colors["NG"]},
            "NG3": {"symbol": "diamond", "color": category_colors["NG"]},
            "NG4": {"symbol": "cross", "color": category_colors["NG"]},

            # Modelos B/P
            "B1": {"symbol": "circle", "color": category_colors["B/P"]},
            "P23": {"symbol": "square", "color": category_colors["B/P"]},
            "P2": {"symbol": "diamond", "color": category_colors["B/P"]},
            "P3": {"symbol": "cross", "color": category_colors["B/P"]},
            "P4": {"symbol": "x", "color": category_colors["B/P"]},

            # Modelos E/R
            "E1": {"symbol": "circle", "color": category_colors["E/R"]},
            "E2": {"symbol": "square", "color": category_colors["E/R"]},
            "R2": {"symbol": "diamond", "color": category_colors["E/R"]},
            "R3": {"symbol": "cross", "color": category_colors["E/R"]}
        }

        # --- Subplot 1: Diferencial normalizada f(α)/f(0.5) ---
        # Añadir curvas teóricas (inicialmente visibles)
        for model in model_attributes:
            if model in teorico_diff:
                attrs = model_attributes.get(model, {"symbol": "circle", "color": "gray"})

                fig.add_trace(
                    go.Scatter(
                        x=alphas_teo,
                        y=teorico_diff[model],
                        name=model,
                        mode='lines+markers',
                        marker=dict(
                            symbol=attrs["symbol"],
                            size=4,
                            color=attrs["color"],
                            line=dict(width=0.5, color='rgba(0,0,0,0.5)')
                        ),
                        line=dict(
                            dash='dash',
                            width=1,
                            color=attrs["color"]
                        ),
                        legendgroup=model,
                        visible=True,
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # Añadir datos experimentales FWO y Starink
        fig.add_trace(
            go.Scatter(
                x=alpha_fwo[mask_fwo],
                y=Z_diff_fwo[mask_fwo],
                name=f'FWO β={beta}',
                mode='lines+markers',
                marker=dict(symbol='circle', size=6, color='black'),
                line=dict(color='black', width=1),
                showlegend=True,
                legendgroup='exp'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=alpha_starink[mask_starink],
                y=Z_diff_starink[mask_starink],
                name=f'Starink β={beta}',
                mode='lines+markers',
                marker=dict(symbol='circle', size=6, color='gray'),
                line=dict(color='gray', width=1),
                showlegend=True,
                legendgroup='exp'
            ),
            row=1, col=1
        )

        # --- Subplot 2: Integral normalizada g(α)/g(0.5) ---
        # Añadir curvas teóricas (inicialmente visibles)
        for model in model_attributes:
            if model in teorico_int:
                attrs = model_attributes.get(model, {"symbol": "circle", "color": "gray"})

                fig.add_trace(
                    go.Scatter(
                        x=alphas_teo,
                        y=teorico_int[model],
                        name=model,
                        mode='lines+markers',
                        marker=dict(
                            symbol=attrs["symbol"],
                            size=4,
                            color=attrs["color"],
                            line=dict(width=0.5, color='rgba(0,0,0,0.5)')
                        ),
                        line=dict(
                            dash='dash',
                            width=1,
                            color=attrs["color"]
                        ),
                        legendgroup=model,
                        visible=True,
                        showlegend=False
                    ),
                    row=1, col=2
                )

        # Añadir datos experimentales FWO y Starink
        fig.add_trace(
            go.Scatter(
                x=alpha_fwo[mask_fwo],
                y=Z_int_fwo[mask_fwo],
                name=f'FWO β={beta}',
                mode='lines+markers',
                marker=dict(symbol='circle', size=6, color='black'),
                line=dict(color='black', width=1),
                showlegend=False,
                legendgroup='exp'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=alpha_starink[mask_starink],
                y=Z_int_starink[mask_starink],
                name=f'Starink β={beta}',
                mode='lines+markers',
                marker=dict(symbol='circle', size=6, color='gray'),
                line=dict(color='gray', width=1),
                showlegend=False,
                legendgroup='exp'
            ),
            row=1, col=2
        )

        # --- Subplot 3: Integral·Diferencial Z(α)=f(α)·g(α) ---
        # Añadir curvas teóricas (inicialmente visibles)
        for model in model_attributes:
            if model in teorico_diff_int:
                attrs = model_attributes.get(model, {"symbol": "circle", "color": "gray"})

                fig.add_trace(
                    go.Scatter(
                        x=alphas_teo,
                        y=teorico_diff_int[model],
                        name=model,
                        mode='lines+markers',
                        marker=dict(
                            symbol=attrs["symbol"],
                            size=4,
                            color=attrs["color"],
                            line=dict(width=0.5, color='rgba(0,0,0,0.5)')
                        ),
                        line=dict(
                            dash='dash',
                            width=1,
                            color=attrs["color"]
                        ),
                        legendgroup=model,
                        visible=True,
                        showlegend=False
                    ),
                    row=1, col=3
                )

        # Añadir datos experimentales FWO y Starink
        fig.add_trace(
            go.Scatter(
                x=alpha_fwo[mask_fwo],
                y=Z_diff_int_fwo[mask_fwo],
                name=f'FWO β={beta}',
                mode='lines+markers',
                marker=dict(symbol='circle', size=6, color='black'),
                line=dict(color='black', width=1),
                showlegend=False,
                legendgroup='exp'
            ),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=alpha_starink[mask_starink],
                y=Z_diff_int_starink[mask_starink],
                name=f'Starink β={beta}',
                mode='lines+markers',
                marker=dict(symbol='circle', size=6, color='gray'),
                line=dict(color='gray', width=1),
                showlegend=False,
                legendgroup='exp'
            ),
            row=1, col=3
        )

        # Layout
        fig.update_layout(
            title=f"Criado's Master Plots: {componente.capitalize()} en {atmosfera.upper()} (β={beta} ºC/min)",
            height=700, width=2400, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, itemsizing='constant'),
            margin=dict(b=100,l=80, r=40),
            plot_bgcolor='rgba(240,240,240,0.9)',
        )

        # Actualizar ejes
        fig.update_xaxes(title_text='α', row=1, col=1)
        fig.update_yaxes(title_text='f(α)/f(0.5)', range=[0, 3.5], row=1, col=1)
        fig.update_xaxes(title_text='α', row=1, col=2)
        fig.update_yaxes(title_text='g(α)/g(0.5)', range=[0, 5], row=1, col=2)
        fig.update_xaxes(title_text='α', row=1, col=3)
        fig.update_yaxes(title_text='Z(α) = f(α)*g(α)', range=[0, 1.5], row=1, col=3)

        # Menú desplegable para mostrar/ocultar modelos teóricos
        buttons = []
        all_models = list(teorico_diff.keys())

        # Botón para mostrar todos los modelos
        buttons.append(dict(label="Todos", method="update", args=[{"visible": [True] * len(fig.data)}]))

        # Botón para ocultar todos los modelos teóricos
        visible_list = [True] * len(fig.data)
        for i, trace in enumerate(fig.data):
            if trace.name in all_models:
                visible_list[i] = False
        buttons.append(dict(label="Ninguno", method="update", args=[{"visible": visible_list}]))

        # Botones individuales para cada modelo
        for model in all_models:
            visible_list = [False] * len(fig.data)
            # Mantener visibles los datos experimentales
            for i, trace in enumerate(fig.data):
                if trace.name not in all_models:  # Datos experimentales
                    visible_list[i] = True
                elif trace.name == model:  # Modelo seleccionado
                    visible_list[i] = True
            buttons.append(dict(label=model, method="update", args=[{"visible": visible_list}]))

        fig.update_layout(
            updatemenus=[dict(type="dropdown", direction="down", buttons=buttons, x=1.0, xanchor="right", y=1.15, yanchor="top")]
        )

        # Guardar gráfico interactivo
        # Creacion de carpetas
        directorio = f"Resultados/F1.1_Criado/{atmosfera.capitalize()}/{componente.capitalize()}"
        os.makedirs(directorio, exist_ok=True)

        # Rutas y nombres
        nombre_base = f"Criado_{componente}_{beta}_{atmosfera}"
        ruta_html = f"{directorio}/{nombre_base}.html"

        # Subida a plotly Chart Studio
        upload_url = py.plot(fig, filename=nombre_base, auto_open=False)

        # Guardado local del HTML
        fig.write_html(ruta_html, include_plotlyjs='cdn')
        ruta_url_txt = f"{directorio}/{nombre_base}_url.txt"
        with open(ruta_url_txt, "w", encoding="utf-8") as f:
            f.write(upload_url)

        print(f"Gráfico interactivo de {nombre_base} guardado")

def fase1_criado():
    """
    Ejecuta el análisis de Criado (master plots) para todas las combinaciones de
    componentes y atmósferas soportadas, guardando los gráficos interactivos.

    Recorre:
        componentes = ['celulosa', 'xilano', 'lignina']
        atmosferas  = ['aire', 'n2']

    Notas:
        - Requiere que `RESULTADOS_GLOBALES['FWO_Starink_Friedman'][comp][atm]['resultados']`
          exista (previamente generado por `fase1_isoconversional`).
    """
    componentes = ['celulosa', 'xilano', 'lignina']
    atmosferas = ['aire', 'n2']

    for componente in componentes:
        for atmosfera in atmosferas:
            try:
                aplicar_criado(componente, atmosfera)
            except Exception as e:
                print(f"Error en {componente}-{atmosfera}: {str(e)}")


# FASE 1.2: APLICAR CR
def ajustar_coats_redfern(temperature_k, alpha, modelo_func, beta):
    """
    Ajusta un modelo cinético por el método integral de Coats–Redfern.

    Implementa la regresión lineal de:
        ln[g(α)/T²] = ln(AR/(βEa)) − Ea/(R·T)

    Para estimar:
        - Energía de activación (Ea)
        - Factor preexponencial (A)
        - Coeficiente de determinación (R²)
        - Parámetros de la recta (pendiente, intercepto)
        - Vectores transformados (x=1/T, y=ln[g(α)/T²]) y el ajuste estimado

    Args:
        temperature_k (np.ndarray): Temperaturas en Kelvin (monótonamente crecientes).
        alpha        (np.ndarray): Fracciones de conversión (0 < α < 1).
        modelo_func  (callable):   Función integral del modelo cinético g(α).
        beta         (float):      Velocidad de calentamiento (K/min).

    Returns:
        tuple:
            Ea (float): Energía de activación [J/mol]
            A (float):  Factor preexponencial [1/min]
            r2 (float): Coeficiente de determinación (0–1)
            pendiente (float): Pendiente de la regresión (= −Ea/R)
            intercepto (float): Intercepto de la regresión
            x (np.ndarray):     1/T [K⁻¹]
            y (np.ndarray):     ln[g(α)/T²]
            ajuste_reg (np.ndarray): Recta ajustada en las x originales

    Notas:
        - Requiere que g(α) sea la forma integral del modelo.
        - Supone mecanismo efectivo invariante en el rango (cinética de un paso).
        - Aproximación válida cuando Ea/(R·T) ≫ 1.
    """
    # Transformación de variables para el ajuste lineal
    t_inv = 1 / temperature_k
    g_alpha = modelo_func(alpha)
    # Variables para la regresión, LinearRegression espera que los datos de entrada x (variables independientes) tengan
    # dos dimensiones, donde cada fila representa un punto de datos y cada columna una variable. Por lo tanto, reshape
    # transforma el vector 1D en un matriz con una sola columna.
    x = t_inv.reshape(-1, 1)
    y = np.log(g_alpha / temperature_k ** 2)

    # Ajuste lineal con sklearn
    reg = LinearRegression()
    reg.fit(x,y)

    # Parámetros del ajuste
    pendiente, intercepto = reg.coef_[0], reg.intercept_
    r2 = reg.score(x, y)

    # Calcular energía de activación (Ea) y factor pre-exponencial (A)
    Ea = -pendiente * R                         # J/mol
    A = (beta * Ea / R) * np.exp(intercepto)    # min^-1

    # Valores predichos por el modelo
    ajuste_reg = pendiente * (1 / temperature_k) + intercepto

    return Ea, A, r2, pendiente, intercepto, x, y, ajuste_reg

def aplicar_modelos_cr(subgrupos, beta):
    """
    Ajusta TODOS los modelos g(α) disponibles (en `g_funcs`) por Coats–Redfern
    en cada subgrupo de datos, y devuelve un diccionario anidado con los resultados.

    Args:
        subgrupos (dict): Estructura con arrays segmentados por rango de T. Debe contener:
                          - 'temp_K': lista de arrays con T (K) por subgrupo
                          - 'alpha' : lista de arrays con α por subgrupo
        beta (float):     Velocidad de calentamiento [K/min] para el cálculo de A.

    Returns:
        dict: {
            "1": { "CR1": {"Ea (J/mol)":..., "A (1/min)":..., "R^2":..., "a":pend, "b":int,
                           "reg_x":x, "reg_y":y, "ajuste_reg":aj }, ...},
            "2": { ... },
            ...
        }

    Notas:
        - Itera sobre todas las claves de `g_funcs` (modelos disponibles).
        - Guarda también los vectores transformados (x, y) y el ajuste para graficar regresiones.
    """
    # Crear un diccionario para almacenar los resultados de los modelos ajustados en cada subgrupo
    resultados = {}
    for i, (temp_K_sub, alpha_sub) in enumerate(zip(subgrupos['temp_K'], subgrupos['alpha'])): # Iterar sobre cada subgrupo y ajustar todos los modelos
        subgrupo_key = f"{i+1}"    # Clave para el subgrupo en el diccionario
        # Crear un sub-diccionario para cada subgrupo
        if subgrupo_key not in resultados:
            resultados [subgrupo_key] = {}

        for nombre, modelo_func in g_funcss.items():
            Ea, A, r2, pendiente, ordenada,x,y,ajuste_reg = ajustar_coats_redfern(temp_K_sub, alpha_sub, modelo_func, beta)
            # Guardar los resultados del modelo dentro del subgrupo
            resultados[subgrupo_key][nombre]={
                "Ea (J/mol)":Ea,
                "A (1/min)": A,
                "R^2": r2,
                "reg_x": x,
                "reg_y": y,
                "ajuste_reg": ajuste_reg,
                "a": pendiente,
                "b": ordenada,
            }
    return resultados

def exportar_parametros_cr_excel(resultados, ruta_archivo):
    """
       Exporta a Excel los parámetros cinéticos (Coats-Redfern) por subgrupo y modelo.

       Toma el diccionario `resultados` devuelto por `aplicar_modelos_cr` (o compatible),
       extrae solo las columnas relevantes y guarda un archivo .xlsx en:
       `Resultados/CoatsRedfern/<subcarpeta>/<nombre_archivo>.xlsx`, donde:
         - `<subcarpeta>` es el último directorio de `ruta_archivo` (p. ej., 'VelocidadCalentamiento15')
         - `<nombre_archivo>` es el nombre base de `ruta_archivo` sin extensión.

       Args:
           resultados (dict): Diccionario anidado con la estructura:
               {
                   "1": {                      # Subgrupo
                       "CR1": {                 # Modelo
                           "Ea (J/mol)": float,
                           "A (1/min)": float,
                           "R^2": float,
                           "a": float,          # Pendiente
                           "b": float,          # Ordenada
                           ...                  # (p. ej. reg_x, reg_y, ajuste_reg)
                       },
                       ...
                   },
                   ...
               }
               Se omiten, si existen, las claves 'reg_x', 'reg_y' y 'ajuste_reg'.

           ruta_archivo (str): Ruta del archivo de origen (p. ej., del .csv/.txt procesado).
               Se usa para:
                 - obtener el nombre base del Excel de salida;
                 - inferir la subcarpeta (velocidad de calentamiento) a partir del directorio padre.

       Efectos:
           - Crea el directorio de salida si no existe: 'Resultados/CoatsRedfern/<subcarpeta>/'.
           - Guarda un Excel con las columnas:
             ['Subgrupo', 'Modelo', 'Ea (J/mol)', 'A (1/min)', 'R2', 'Pendiente', 'Ordenada'].
             Nota: 'R2' en el Excel proviene de la clave 'R^2' del diccionario.
           - Imprime por pantalla la ruta completa del archivo guardado.

       Returns:
           None

       Notas:
           - Requiere tener importados `pandas as pd` y `os`.
           - `nombre_archivo` es el nombre de `ruta_archivo` sin extensión (p. ej., 'celulosa_15_aire').
       """

    # Convertir el diccionario de resultados a un Dataframe
    resultados_list =[] # Creacion lista vacia

    for subgrupo,modelos in resultados.items():
        for modelo, valores in modelos.items():
            # Creacion de una fila solo con las columnas necesarias (omitiendo en este caso reg_x, reg_y y ajuste_reg
            fila = {
                "Subgrupo": subgrupo,
                "Modelo": modelo,
                "Ea (J/mol)": valores["Ea (J/mol)"],
                "A (1/min)": valores["A (1/min)"],
                "R2": valores["R^2"],
                "Pendiente":valores ["a"],
                "Ordenada": valores ["b"]
            }
            resultados_list.append(fila)

    # Convertir la lista de diccionarios en un Dataframe
    df_resultados = pd.DataFrame(resultados_list)

    # Obtener el nombre del archivo sin la extensión (ejemplo: 'celulosa_15_aire')
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]

    # Determinar la subcarpeta (VelocidadCalentamiento5, VelocidadCalentamiento15, VelocidadCalentamiento30)
    subcarpeta = os.path.dirname(ruta_archivo).split(os.sep)[-1]

    # Crear la ruta del directorio donde se guardará el archivo de Excel
    ruta_directorio_salida = os.path.join('Resultados', 'F1.2_CoatsRedfern', subcarpeta)

    # Crear el directorio si no existe
    os.makedirs(ruta_directorio_salida, exist_ok=True)

    # Crear el nombre del archivo Excel (por ejemplo: 'celulosa_15_aire.xlsx')
    nombre_archivo_excel = f'{nombre_archivo}.xlsx'

    # Ruta completa del archivo Excel
    ruta_salida = os.path.join(ruta_directorio_salida, nombre_archivo_excel)

    # Guardar el DataFrame en un archivo Excel
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en '{ruta_salida}'")

def representar_regresion(resultados, ruta_archivo):
    """
    Genera y guarda figuras de regresión (2×4 subplots por figura) para todos los modelos
    ajustados por Coats–Redfern, separadas por subgrupo.

    Args:
        resultados (dict): Estructura anidada devuelta por el ajuste (por subgrupo y modelo).
        ruta_archivo (str): Ruta del archivo original; se usa para el nombre base y subcarpeta.

    Efectos:
        - Crea PNGs en: Resultados/Regresiones/<VelocidadCalentamientoX>/<nombre>_sub<k>_parte<i>.png
        - Cada subplot muestra datos transformados (x=1/T, y=ln[g(α)/T²]) y su recta ajustada,
          junto con Ea (kJ/mol), A (1/min) y R² en el título.
    """

    nombre_componente = os.path.splitext(os.path.basename(ruta_archivo))[0]

    # Listado de modelos en orden consistente
    modelos = list(next(iter(resultados.values())).keys())  # e.g. ['CR0','CR05',...,'R3']
    # Partir en trozos de 8 modelos
    trozos = [modelos[i:i+8] for i in range(0, len(modelos), 8)]

    for subgrupo_key, subgrupo_val in resultados.items():
        # Para cada “trozo” de 8 modelos, crear una figura 2x4
        for fig_idx, chunk in enumerate(trozos, start=1):
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            axs = axs.flatten()

            for ax, nombre in zip(axs, chunk):
                datos = subgrupo_val[nombre]
                x = datos["reg_x"]
                y = datos["reg_y"]
                ajuste = datos["ajuste_reg"]
                Ea = datos["Ea (J/mol)"] / 1000  # pasar a kJ/mol
                A  = datos["A (1/min)"]
                R2 = datos["R^2"]

                ax.plot(x, y, '--', label='datos')
                ax.plot(x, ajuste, '-', label=f'Ajuste {nombre}')
                ax.set_xlabel('1/T (K⁻¹)')
                ax.set_ylabel('ln(g(α)/T²)')
                ax.set_title(
                    f'{nombre}\nEa={Ea:.1f} kJ/mol  A={A:.1e}  R²={R2:.3f}',
                    fontsize='large'
                )
                ax.legend(fontsize='large')
                ax.tick_params(axis='x', rotation=45)

                temp_inicial = int((1 / x[0]) - 273)
                temp_final = int((1 / x[-1]) - 273)

            # Si quedan ejes sin modelo, se apagan
            for ax in axs[len(chunk):]:
                ax.axis('off')

            fig.tight_layout()
            fig.suptitle(
                f"{nombre_componente} ({temp_inicial}ºC - {temp_final}ºC) — Subgrupo {subgrupo_key}  (Fragmento {fig_idx}/{len(trozos)})",
                y=1.02, fontsize=16
            )

            # Guardado
            subcarpeta = os.path.dirname(ruta_archivo).split(os.sep)[-1]
            out_dir = f'Resultados/F1.2_RegresionesCR/{subcarpeta}'
            os.makedirs(out_dir, exist_ok=True)
            ruta_out = os.path.join(out_dir, f'{nombre_componente}_sub{subgrupo_key}_parte{fig_idx}.png')
            plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Grafico regresiones guardado en: {ruta_out}")

def fase1_cr():
    """
        Ejecuta en lote el ajuste por Coats–Redfern sobre los datos de `DATOS_PROCESADOS["DatosComponentes"]`.

        Flujo por cada combinación <VelocidadCalentamientoX>/<componente>:
            1) Divide en subgrupos (según temperaturas seleccionadas).
            2) Ajusta todos los modelos por Coats–Redfern en cada subgrupo.
            3) Exporta parámetros a Excel.
            4) Genera figuras de regresión.

        Efectos:
            - Excel en  Resultados/CoatsRedfern/<VelocidadCalentamientoX>/<nombre>.xlsx
            - Gráficas en Resultados/Regresiones/<VelocidadCalentamientoX>/<nombre>_sub<i>_parte<j>.png
        """
    for velocidad_key, componentes in DATOS_PROCESADOS["DatosComponentes"].items():
        beta = int(velocidad_key.replace("VelocidadCalentamiento", ""))

        for nombre_componente, datos in componentes.items():
            try:
                # Obtener datos procesados del componente
                temps_seleccionadas = datos['temp_seleccionadas']
                temperature = datos['temperature']
                temperature_k = datos['temperature_k']
                alpha = datos['alpha']
                weight_mg = datos['weight_mg']
                time = datos['time']
                heat_flow_q = datos['heat_flow_q']


                # Dividir en subgrupos según temperaturas seleccionadas
                subgrupos, indices = dividir_en_subgrupos(
                    temperature, temperature_k, alpha, weight_mg, time, heat_flow_q, temps_seleccionadas
                )

                # Aplicar modelos cinéticos a los subgrupos
                resultados = aplicar_modelos_cr(subgrupos, beta)

                # Guardar resultados en Excel
                ruta_archivo = f"{velocidad_key}/{nombre_componente}.csv"  # Ruta simulada para guardado
                exportar_parametros_cr_excel(resultados, ruta_archivo) #De momento al aplicar representar_regresion2

                # Generar gráficos de regresión
                representar_regresion(resultados, ruta_archivo)

                print(f"Procesado: {nombre_componente} ({velocidad_key})")

            except Exception as e:
                print(f"Error en {nombre_componente}: {str(e)}")


# FASE 2.0. RECONSTRUCCIÓN COMPONENTES PUROS
def get_celulosa_model_dadT(beta=15):
    """
    Devuelve la función ODE dα/dT para la celulosa y la condición inicial α0, para una β dada.

    Args:
        beta (float): Velocidad de calentamiento [°C/min] empleada en la ODE.

    Returns:
        tuple[callable, float]: (dadT_total, alpha0)
            - dadT_total(T, α): función que calcula dα/dT en función de la temperatura T (K) y la conversión α (–).
            - alpha0 (float): valor inicial de conversión (–).

    Notas:
        - Incluye una fase de humedad lineal hasta 100 °C (373 K).
        - Considera dos mecanismos separados por T_switch (350 °C).
        - Requiere constantes/funciones globales: R, ng15_f, cr4_f.
    """
    # Fase de secado de la humedad
    alpha0 = 0.0
    alpha_moisture = 0.25
    T_moisture_end = 100 + 273

    # Parámetros cinéticos
    A1, Ea1 = 3e11, 138e3
    A2, Ea2 = 2e15, 180e3
    T_switch = 350 + 273

    def dadT_total(T, alpha):
        if T <= T_moisture_end:
            return alpha_moisture / T_moisture_end
        elif T <= T_switch:
            return (A1 / beta) * np.exp(-Ea1 / (R * T)) * ng15_f(alpha)
        else:
            return (A2 / beta) * np.exp(-Ea2 / (R * T)) * cr4_f(alpha)      # Se ha decantado por este modelo, tras hacer varias simulaciones.

    return dadT_total, alpha0

def get_xilano_model_dadT(beta=15):
    """
    Devuelve la función ODE dα/dT para el xilano y la condición inicial α0, para una β dada.

    Args:
        beta (float): Velocidad de calentamiento [°C/min] empleada en la ODE.

    Returns:
        tuple[callable, float]: (dadT_total, alpha0)
            - dadT_total(T, α): función que calcula dα/dT en función de T (K) y α (–).
            - alpha0 (float): valor inicial de conversión (–).

    Notas:
        - Incluye fase de humedad lineal hasta 100 °C (373 K).
        - Usa el mismo modelo cinético (cr15_f) antes y después de T_switch (285 °C).
        - Limita la reacción cuando α > 0.95 (término de corte).
        - Requiere constantes/funciones globales: R, cr15_f.
    """

    alpha0 = 0.0
    alpha_moisture = 0.30
    T_moisture_end = 100 + 273

    A1, Ea1 = 9.8e13, 148e3
    A2, Ea2 = 2e2, 40e3
    T_switch = 285 + 273

    def dadT_total(T, alpha):
        if alpha > 0.95:
            return 0.0
        if T <= T_moisture_end:
            return alpha_moisture / T_moisture_end
        elif T <= T_switch:
            return (A1 / beta) * np.exp(-Ea1 / (R * T)) * cr15_f(alpha)
        else:
            return (A2 / beta) * np.exp(-Ea2 / (R * T)) * cr15_f(alpha)

    return dadT_total, alpha0

def get_lignina_model_dadT(beta=15):
    """
    Devuelve la función ODE dα/dT para la lignina y la condición inicial α0, para una β dada.

    Args:
        beta (float): Velocidad de calentamiento [°C/min] empleada en la ODE.

    Returns:
        tuple[callable, float]: (dadT_total, alpha0)
            - dadT_total(T, α): función que calcula dα/dT en función de T (K) y α (–).
            - alpha0 (float): valor inicial de conversión (–).

    Notas:
        - Incluye fase de humedad lineal hasta 100 °C (373 K).
        - Considera un único mecanismo cinético (DM1).
        - Limita la reacción cuando α > 0.96.
        - Requiere constantes/funciones globales: R, dm1_f.
    """
    alpha0 = 0.0
    alpha_moisture = 0.40
    T_moisture_end = 100 + 273

    A1, Ea1 = 3e1, 38e3

    def dadT_total(T, alpha):
        if alpha > 0.96:
            return 0.0
        if T <= T_moisture_end:
            return alpha_moisture / T_moisture_end
        else:
            return (A1 / beta) * np.exp(-Ea1 / (R * T)) * dm1_f(alpha)

    return dadT_total, alpha0

def reconstruccion_celulosa():
    """
    Reconstruye α(T) y dα/dT(T) para celulosa (β=15 °C/min, aire) resolviendo la ODE
    de descomposición con fase de humedad y dos mecanismos. Compara con los datos
    experimentales y guarda una figura con dos subplots: α(T) y dα/dT(T).

    Datos:
        DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['celulosa_15_aire']
        - 'temperature_k' (K), 'alpha' (–)

    Notas:
        - Obtiene la ODE desde get_celulosa_model_dadT(beta=15).
        - Requiere: R, np, plt, solve_ivp, os, DATOS_PROCESADOS.
        - Guarda en: Resultados/ReconstruccionMuestras/Celulosa/.
    """
    # ODE y condición inicial desde el getter
    dadT_total, alpha0 = get_celulosa_model_dadT(beta=15)

    # Datos experimentales
    T_exp = np.array(DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['celulosa_15_aire']['temperature_k'])
    alpha_exp = np.array(DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['celulosa_15_aire']['alpha'])

    # Ordenar por T
    sort_idx = np.argsort(T_exp)
    T_exp = T_exp[sort_idx]
    alpha_exp = alpha_exp[sort_idx]

    # Resolver ODE para α(T)
    sol = solve_ivp(fun = lambda T, a:dadT_total(T, a[0]),
                    t_span=(T_exp.min(), T_exp.max()),
                    y0=[0.0], t_eval=T_exp, method='RK45')
    alpha_model = sol.y[0]

    # Derivadas
    dalpha_dT_exp = np.gradient(alpha_exp, T_exp)
    dalpha_dT_model = np.gradient(alpha_model, T_exp)

    # Figura
    plt.figure(figsize=(12, 5))

    # α(T)
    plt.subplot(1, 2, 1)
    plt.plot(T_exp - 273, alpha_exp, 'o', markersize=1.5, label='Experimental', color = 'lightgray')
    plt.plot(T_exp - 273, alpha_model, '-', label='Modelo (ODE)')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Conversión α')
    plt.title('Curva de Conversión: α(T)')
    plt.legend()

    # dα/dT(T)
    plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, posición 2
    plt.plot(T_exp - 273, dalpha_dT_exp, 'o', markersize=1.5, label='dα/dT Experimental', color = 'lightgray')
    plt.plot(T_exp - 273, dalpha_dT_model, '-', label='dα/dT Modelado')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('dα/dT [1/°C]')
    plt.title('Velocidad de Conversión')
    plt.legend()

    plt.suptitle('Reconstrucción — Celulosa (β=15 °C/min, aire)', fontsize=14, y=1.0)

    # Guardado
    out_dir = os.path.join("Resultados", "F2.0_ReconstruccionComponentes", "Celulosa")
    os.makedirs(out_dir, exist_ok=True)
    ruta_out = os.path.join(out_dir, "Reconstruccion_Celulosa_15_aire.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reconstrucción de Celulosa guardada en: {ruta_out}")

def reconstruccion_xilano():
    """
    Reconstruye α(T) y dα/dT(T) para xilano (β=15 °C/min, aire) con fase de humedad
    y cambio de mecanismo en T_switch. Limita la conversión a α ≤ 0.95. Compara con
    experimental y guarda α(T) y dα/dT(T).

    Datos:
        DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['xilano_15_aire']

    Notas:
        - ODE desde get_xilano_model_dadT(beta=15).
        - Ajusta T_eval para garantizar monotonía estricta antes de integrar.
        - Guarda en: Resultados/ReconstruccionMuestras/Xilano/.
    """
    # ODE y condición inicial
    dadT_total, alpha0 = get_xilano_model_dadT(beta=15)

    # Datos experimentales
    T_exp = np.array(DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['xilano_15_aire']['temperature_k'])
    alpha_exp = np.array(DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['xilano_15_aire']['alpha'])

    # Ordenar
    sort_idx = np.argsort(T_exp)
    T_exp_sorted = T_exp[sort_idx]
    alpha_exp_sorted = alpha_exp[sort_idx]

    # Asegurar monotonía estricta en T para el integrador
    T_exp_adj = T_exp_sorted.copy().astype(float)
    for i in range(1, len(T_exp_adj)):
        if T_exp_adj[i] <= T_exp_adj[i - 1]:
            # Añadir incremento proporcional al índice para mantener orden
            T_exp_adj[i] = T_exp_adj[i - 1] + 1e-4 * (i / len(T_exp_adj))

    t_span = (T_exp.min(), T_exp.max())
    sol = solve_ivp(dadT_total, t_span, y0=[0.0], t_eval=T_exp_adj, method='RK45')
    alpha_model = sol.y[0]

    # Derivadas
    dalpha_dT_exp = np.gradient(alpha_exp_sorted, T_exp)
    dalpha_dT_model = np.gradient(alpha_model, T_exp_sorted)

    # Figura
    plt.figure(figsize=(12,5))  # Ancho aumentado para acomodar ambos gráficos

    # α(T)
    plt.subplot(1, 2, 1)
    plt.plot(T_exp - 273, alpha_exp, 'o', markersize=1.5, label='Experimental', color = 'lightgray')
    plt.plot(T_exp - 273, alpha_model, '-', label='Modelo (ODE)')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Conversión α')
    plt.title('Curva de Conversión: α(T)')
    plt.legend()

    # dα/dT(T)
    plt.subplot(1, 2, 2)
    plt.plot(T_exp - 273, dalpha_dT_exp, 'o', markersize=1.5, label='dα/dT Experimental', color = 'lightgray')
    plt.plot(T_exp - 273, dalpha_dT_model, '-', label='dα/dT Modelado')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('dα/dT [1/°C]')
    plt.title('Velocidad de Conversión')
    plt.legend()

    plt.suptitle('Reconstrucción — Xilano (β=15 °C/min, aire)', fontsize=14, y=1.0)

    # Guardado
    out_dir = os.path.join("Resultados", "F2.0_ReconstruccionComponentes", "Xilano")
    os.makedirs(out_dir, exist_ok=True)
    ruta_out = os.path.join(out_dir, "Reconstruccion_Xilano_15_aire.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reconstrucción de Xilano guardada en: {ruta_out}")

def reconstruccion_lignina():
    """
    Reconstruye α(T) y dα/dT(T) para lignina (β=15 °C/min, aire) con fase de humedad
    y un único mecanismo (DM1). Limita la conversión a α ≤ 0.96. Fija el eje y de
    dα/dT a (0, 0.02) y guarda la figura.

    Datos:
        DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['lignina_15_aire']

    Notas:
        - ODE desde get_lignina_model_dadT(beta=15).
        - Garantiza monotonía estricta de T antes de integrar.
        - Guarda en: Resultados/ReconstruccionMuestras/Lignina/.
    """
    # ODE y condición inicial
    dadT_total, alpha0 = get_lignina_model_dadT(beta=15)

    # Datos experimentales
    T_exp = np.array(DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['lignina_15_aire']['temperature_k'])
    alpha_exp = np.array(DATOS_PROCESADOS['DatosComponentes']['VelocidadCalentamiento15']['lignina_15_aire']['alpha'])

    # Ordenar
    sort_idx = np.argsort(T_exp)
    T_exp_sorted = T_exp[sort_idx]
    alpha_exp_sorted = alpha_exp[sort_idx]

    # Monotonía estricta
    T_exp_adj = T_exp_sorted.copy().astype(float)
    for i in range(1, len(T_exp_adj)):
        if T_exp_adj[i] <= T_exp_adj[i - 1]:
            # Añadir incremento proporcional al índice para mantener orden
            T_exp_adj[i] = T_exp_adj[i - 1] + 1e-4 * (i / len(T_exp_adj))

    t_span = (T_exp.min(), T_exp.max())
    sol = solve_ivp(dadT_total, t_span, y0=[0.0], t_eval=T_exp_adj, method='RK45')
    alpha_model = sol.y[0]

    # Derivadas
    dalpha_dT_exp = np.gradient(alpha_exp_sorted, T_exp)
    dalpha_dT_model = np.gradient(alpha_model, T_exp_sorted)

    # Figura
    plt.figure(figsize=(12, 5))

    # α(T)
    plt.subplot(1, 2, 1)
    plt.plot(T_exp - 273, alpha_exp, 'o', markersize=1.5, label='Experimental', color = 'lightgray')
    plt.plot(T_exp - 273, alpha_model, '-', label='Modelo (ODE)')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Conversión α')
    plt.title('Curva de Conversión: α(T)')
    plt.legend()

    # dα/dT(T)
    plt.subplot(1, 2, 2)
    plt.plot(T_exp - 273, dalpha_dT_exp, 'o', markersize=1.5, label='dα/dT Experimental', color = 'lightgray')
    plt.plot(T_exp - 273, dalpha_dT_model, '-', label='dα/dT Modelado')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('dα/dT [1/°C]')
    plt.title('Velocidad de Conversión')
    plt.legend()
    plt.ylim(0, 0.02)

    plt.suptitle('Reconstrucción — Lignina (β=15 °C/min, aire)', fontsize=14, y=1.0)

    # Guardado
    out_dir = os.path.join("Resultados", "F2.0_ReconstruccionComponentes", "Lignina")
    os.makedirs(out_dir, exist_ok=True)
    ruta_out = os.path.join(out_dir, "Reconstruccion_Lignina_15_aire.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reconstrucción de Xilano guardada en: {ruta_out}")

# FASE 2.1: RECONSTRUCCIÓN MUESTRAS DE COMPOSICIÓN CONOCIDA
def reconstruir_muestra_unica(nombre_muestra, pesos, beta=15):
    """
    Reconstruye α(T) y dα/dT(T) de una muestra como combinación ponderada de
    celulosa, xilano y lignina, ajustada a la rampa de T de la muestra.
    Genera y guarda una figura con α(T) en el eje izquierdo y dα/dT(T) en el derecho.

    Args:
        nombre_muestra (str): Clave de la muestra en DATOS_PROCESADOS, p.ej. 'muestra1'...'muestra5'.
        pesos (list[float]): [w_celulosa, w_xilano, w_lignina]. Deben sumar ≈ 1.
        beta (int|float): Velocidad de calentamiento [°C/min] (por defecto 15).

    Efectos:
        - Lee de: DATOS_PROCESADOS['DatosMuestras'][f'VelocidadCalentamiento{beta}'][nombre_muestra]
            -> 'temperature_k', 'alpha'
        - Usa ODEs de: get_celulosa_model_dadT, get_xilano_model_dadT, get_lignina_model_dadT
        - Guarda PNG en: Resultados/ReconstruccionMuestras/<NombreCarpeta>/
            donde <NombreCarpeta> = 'MuestraN' según nombre_muestra.
    """
    # --- Cargar datos de la muestra ---
    clave_beta = f'VelocidadCalentamiento{beta}'
    datos = DATOS_PROCESADOS['DatosMuestras'][clave_beta][nombre_muestra]
    T_K = np.array(datos['temperature_k'])
    alpha_exp = np.array(datos['alpha'])

    # Ordenar por T (K)
    idx = np.argsort(T_K)
    T_K = T_K[idx]
    alpha_exp = alpha_exp[idx]

    # Asegurar monotonía estricta de T para evaluar la ODE
    T_eval = T_K.copy().astype(float)
    for i in range(1, len(T_eval)):
        if T_eval[i] <= T_eval[i - 1]:
            T_eval[i] = T_eval[i - 1] + 1e-4 * (i / len(T_eval))

    # --- Obtener modelos de componentes ---
    modelos = [get_celulosa_model_dadT(beta), get_xilano_model_dadT(beta), get_lignina_model_dadT(beta)]
    if len(pesos) != 3:
        raise ValueError("Los 'pesos' deben ser una lista de 3 elementos: [celulosa, xilano, lignina].")

    # Resolver ODE de cada componente en la rampa de la muestra
    alpha_components = []
    for (dadT, alpha0), w in zip(modelos, pesos):
        sol = solve_ivp(fun=lambda T, a: dadT(T, a[0]),
                        t_span=(T_eval.min(), T_eval.max()),
                        y0=[alpha0],
                        t_eval=T_eval,
                        method='RK45')
        alpha_components.append(sol.y[0])

    # Combinación ponderada
    alpha_total = sum(w * comp for w, comp in zip(pesos, alpha_components))

    # Derivadas (usar T_K para consistencia de ejes)
    dadt_model = np.gradient(alpha_total, T_K)
    dadt_exp = np.gradient(alpha_exp, T_K)
    dadt_exp = suavizar_dtg(dadt_exp, 100)

    # Conversión a °C solo para representación
    T_C = T_K - 273

    # --- Gráfica combinada (α y dα/dT) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # α(T)
    ax1.plot(T_C, alpha_exp, '--', color='#bfbfbf', label='Muestra exp.', markersize=0.5)
    ax1.plot(T_C, alpha_total, '-', color='#000000', label='Muestra modelada')
    ax1.set_xlabel('Temperatura (°C)', fontsize=12)
    ax1.set_ylabel('Conversión α', fontsize=12)

    # Eje derecho para dα/dT
    ax2 = ax1.twinx()
    ax2.plot(T_C, dadt_exp, '--', color='#b7c8da', label='dα/dT exp.', markersize=0.5)
    ax2.plot(T_C, dadt_model, '-', color='#003366', label='dα/dT modelado')
    ax2.set_ylabel('dα/dT', fontsize=12)

    # Combinar leyendas
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)

    # Título
    wC, wX, wL = pesos
    nombre_pretty = nombre_muestra.title()  # 'muestra5' -> 'Muestra5'
    plt.title(f'Reconstrucción de {nombre_pretty} (C:{wC:.3f}, X:{wX:.3f}, L:{wL:.3f})', fontsize=12)
    plt.tight_layout()

    # --- Guardado ---
    out_dir = os.path.join("Resultados", "F2.1_ReconstruccionMuestras", nombre_pretty)
    os.makedirs(out_dir, exist_ok=True)
    ruta_out = os.path.join(out_dir, f"Reconstruccion_{nombre_pretty}_{beta}_aire.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Figura guardada: {ruta_out}")

def reconstruccion_muestras(beta=15):
    """
    Reconstruye y guarda las figuras de las 5 muestras: muestra1..muestra5.

    Args:
        beta (int|float): Velocidad de calentamiento [°C/min]. Por defecto 15.

    Notas:
        - Usa pesos conocidos para muestra5 (C=0.493, X=0.227, L=0.280).
        - Para las demás muestras, aplica pesos por defecto [1/3, 1/3, 1/3].
          Si tienes composiciones específicas, añádelas en 'pesos_por_muestra'.
    """
    muestras = [f"muestra{i}" for i in range(1, 6)]

    # Definición composición de las muestras
    pesos_por_muestra = {
        "muestra1": [0.518, 0.0, 0.482],
        "muestra2": [0.528, 0.472, 0.0],
        "muestra3": [0.0, 0.471, 0.529],
        "muestra4": [0.304, 0.377, 0.319],
        "muestra5": [0.493, 0.227, 0.280],
    }

    for m in muestras:
        pesos = pesos_por_muestra.get(m)
        reconstruir_muestra_unica(m, pesos, beta=beta)

# FASE 2.2: SIMULACIÓN DE MUESTRA 5 BAJO DISTINTAS BETAS
def simulacion_muestra_betas():
    """
    Simula la descomposición térmica de una muestra (por defecto, 'muestra5') bajo
    múltiples velocidades de calentamiento (β = 5, 10, 15, 20, 30 °C/min) y
    compara las curvas modeladas de α(T) y dα/dT(T).

    Descripción:
        - Usa la rampa de T de la muestra desde:
          DATOS_PROCESADOS['DatosMuestras']['VelocidadCalentamiento15']['muestra5']['temperature_k']
        - Integra las ODEs dα/dT de celulosa, xilano y lignina para varias β con:
          get_celulosa_model_dadT, get_xilano_model_dadT, get_lignina_model_dadT.
        - Combina las conversiones con pesos (por defecto C=0.493, X=0.227, L=0.280).
        - Calcula dα/dT numérico y dibuja:
            (1) dα/dT vs T(°C)   (2) α vs T(°C).
        - **Guarda** la figura en: Resultados/SimulacionMuestraBetas/Simulacion_Muestra5_multibeta.png

    Notas:
        - Para usar otra muestra/pesos, modifica la clave 'muestra5' y el vector 'weights'.
        - Se fuerza monotonía estricta en T para la integración añadiendo un pequeño
          incremento cuando hay empates.
        - Requiere: numpy as np, matplotlib.pyplot as plt, scipy.integrate.solve_ivp,
          os, y los getters de componente (get_*_model).
    """
    betas = [5,10,15,20,30]

    # Eje x (rampa de la muestra seleccionada)
    T_muestra = np.array(DATOS_PROCESADOS['DatosMuestras']['VelocidadCalentamiento15']['muestra5']['temperature_k'])

    # Ordenar
    idx = np.argsort(T_muestra)
    T_muestra = T_muestra[idx]

    # Asegurar monotonía estricta en T para el integrador
    T_exp_adj = T_muestra.copy().astype(float)
    for i in range(1, len(T_exp_adj)):
        if T_exp_adj[i] <= T_exp_adj[i - 1]:
            # Añadir incremento proporcional al índice para mantener orden
            T_exp_adj[i] = T_exp_adj[i - 1] + 1e-4 * (i / len(T_exp_adj))

    # Pesos (C, X, L) — ajustar según la muestra si procede
    weights = [0.493, 0.227, 0.28]

    # Estructura auxiliar (solo para graficar; no se devuelve)
    results = {
        'T': T_exp_adj - 273,  # Convertir a °C para graficar
        'alpha_total': {},
        'dalpha_dT': {}
    }

    plt.figure(figsize=(8, 6))

    for beta in betas:
        alpha_components = []
        models = [
            get_celulosa_model_dadT(beta),
            get_xilano_model_dadT(beta),
            get_lignina_model_dadT(beta)
        ]

        # Resolver cada componente en la rampa de la muestra
        for (dadT, alpha0), w in zip(models, weights):
            sol = solve_ivp(
                fun=lambda T, a: dadT(T, a[0]),
                t_span=(T_exp_adj.min(), T_exp_adj.max()),
                y0=[alpha0],
                t_eval=T_exp_adj,
                method='RK45',
                vectorized=True
            )
            alpha_components.append(sol.y[0])

        # Conversión total ponderada
        alpha_total = sum(w * comp for w, comp in zip(weights, alpha_components))
        results['alpha_total'][beta] = alpha_total

        # Derivada numérica respecto a T (K)
        dalpha_dT = np.gradient(alpha_total, T_muestra)
        results['dalpha_dT'][beta] = dalpha_dT

        # Graficar
        plt.subplot(2, 1, 1)
        plt.plot(results['T'], dalpha_dT, label=f'{beta}°C/min')

        plt.subplot(2, 1, 2)
        plt.plot(results['T'], alpha_total, label=f'{beta}°C/min')

    # Configurar gráfica dalpha/dT
    plt.subplot(2, 1, 1)
    plt.xlabel('Temperatura (°C)', fontsize = 10)
    plt.ylabel('dα/dT', fontsize = 10)
    plt.title('Velocidad de Conversión vs. Temperatura', fontsize = 12)
    plt.legend()

    # Configurar gráfica α
    plt.subplot(2, 1, 2)
    plt.xlabel('Temperatura (°C)', fontsize = 10)
    plt.ylabel('α', fontsize = 10)
    plt.title('Conversión vs. Temperatura', fontsize = 12)
    plt.legend()

    plt.tight_layout()

    # Guardado en disco (en lugar de mostrar)
    out_dir = os.path.join("Resultados", "F2.2_SimulacionMuestraBetas")
    os.makedirs(out_dir, exist_ok=True)
    ruta_out = os.path.join(out_dir, "Simulacion_Muestra5_multibeta.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figura guardada en: {ruta_out}")

# FASE 2.3: SIMULACIÓN COMBUSTIÓN MUESTRA 5 BAJO EVOLUCIÓN DE TEMPERATURA "REAL"
def iso834_modificada(t):
    """
    Perfil de temperatura ISO 834 **modificada con ayuda de ChatGPT** para incendios de biomasa.

    Estructura del perfil:
        1) Crecimiento exponencial inicial hasta T_max en ~t_crecimiento.
        2) Meseta a T_max hasta t_meseta.
        3) Enfriamiento exponencial con constante de tiempo τ.

    Args:
        t (float): Tiempo [min].

    Returns:
        tuple[float, float]: (T, dT/dt), con T en Kelvin y dT/dt en K/min.

    Notas:
        - dT/dt puede emplearse como β(t) para convertir dα/dT → dα/dt.
        - Esta forma es una adaptación inspirada en ISO 834 ajustada empíricamente.
    """
    # Parámetros del modelo
    T_amb = 25 + 273        # Temperatura ambiente en K (25°C)
    T_max = 800 + 273       # Temperatura máxima (700°C)
    t_crecimiento = 10      # Tiempo para alcanzar T_max (min)
    t_meseta = 30           # Duración de la fase de meseta (min)
    tau = 15                # Constante de tiempo para enfriamiento (min) --> Cuanto mas pequeño, antes enfriara el sistema

    # Fase de crecimiento (0–t_crecimiento)
    if t <= t_crecimiento:
        k = -np.log(0.05) / t_crecimiento  # 95% de T_max en t_crecimiento. Como aqui no pienso representar la Temperatura, no hare el ajuste ese que hice en colab
        T = T_amb + (T_max - T_amb) * (1 - np.exp(-k * t))
        dTdt = k * (T_max - T_amb) * np.exp(-k * t)

    # Fase de meseta (t_crecimiento–t_meseta)
    elif t <= t_meseta:
        T = T_max
        dTdt = 0
    # Fase de enfriamiento (t > t_meseta)
    else:
        t0 = t_meseta
        T = T_amb + (T_max - T_amb) * np.exp(-(t - t0) / tau)
        dTdt = -(T_max - T_amb) / tau * np.exp(-(t - t0) / tau)
    return T,dTdt

def curva_exponencial(t):
    """
    Curva de temperatura exponencial para incendios de biomasa.

    Descripción del perfil:
        - Crecimiento (0 → t_growth):
            T = T_amb + (T_max - T_amb) * (t/t_growth)^2
            dT/dt = 2*(T_max - T_amb)*t / t_growth^2
        - Enfriamiento (t > t_growth):
            τ = τ_base + 0.5*(t - t_growth)
            T = T_amb + (T_max - T_amb) * exp(-(t - t_growth)/τ)
            dT/dt = (T_max - T_amb) * exp_term * [(t - t_growth)/τ^2 - 1/τ]

    Args:
        t (float): Tiempo [min].

    Returns:
        tuple[float, float]: (T, dT/dt), con T en Kelvin y dT/dt en K/min.
    """
    T_amb = 25 + 273.15  # Temperatura ambiente en K
    T_max = 700 + 273.15  # Temperatura máxima en K
    t_growth = 15  # Tiempo para alcanzar T_max (min)
    tau_base = 25  # Constante de tiempo base (min)

    # Fase de crecimiento (0 - t_growth min)
    if t <= t_growth:
        # T = T_amb + (T_max - T_amb) * (t/t_growth)^2
        T = T_amb + (T_max - T_amb) * (t / t_growth) ** 2
        dTdt = 2 * (T_max - T_amb) * t / t_growth ** 2
    # Fase de enfriamiento (t > t_growth)
    else:
        # Enfriamiento con tau creciente: τ = τ_base + 0.5*(t - t_growth)
        tau = tau_base + 0.5 * (t - t_growth)
        exp_term = np.exp(-(t - t_growth) / tau)
        T = T_amb + (T_max - T_amb) * exp_term

        # Derivada: dT/dt = (T_max - T_amb) * exp_term * [(t - t_growth)/tau^2 - 1/tau]
        dTdt = (T_max - T_amb) * exp_term * ((t - t_growth) / (tau ** 2) - 1 / tau)

    return T, dTdt

def curva_logaritmica(t):
    """
    Curva de temperatura logarítmica para incendios de biomasa.

    Descripción del perfil:
        - Fase de crecimiento (0 → 20 min):
            T = T_amb + (T_max - T_amb) * (1 - exp(-spread_rate * t))
            dT/dt = (T_max - T_amb) * spread_rate * exp(-spread_rate * t)
        - Fase de enfriamiento (t > 20 min), con τ variable decreciente:
            T20 = T_amb + (T_max - T_amb) * (1 - exp(-spread_rate * 20))
            τ   = 35 * (1 - 0.7 * min(1, (t - 20) / 40))
            T   = T_amb + (T20 - T_amb) * exp(-(t - 20)/τ)
            dT/dt = -(T20 - T_amb)/τ * exp_term * [1 + (t-20)/(τ) * 0.7/40]

    Args:
        t (float): Tiempo [min].

    Returns:
        tuple[float, float]: (T, dT/dt), con T en Kelvin y dT/dt en K/min.
    """
    T_amb = 25 + 273.15  # Temperatura ambiente en K
    T_max = 700 + 273.15  # Temperatura máxima en K
    I_R = 2000  # Intensidad de radiación (kW/m²)
    ρ = 500  # Densidad del combustible (kg/m³) - Valor medio

    # Tasa de propagación del fuego (m/min)
    spread_rate = 0.05 * I_R / (ρ * 1.8)

    # Fase de crecimiento (0-20 min)
    if t <= 20:
        T = T_amb + (T_max - T_amb) * (1 - np.exp(-spread_rate * t))
        dTdt = (T_max - T_amb) * spread_rate * np.exp(-spread_rate * t)

    # Fase de enfriamiento (t > 20 min)
    else:
        # Temperatura al inicio del enfriamiento
        T20 = T_amb + (T_max - T_amb) * (1 - np.exp(-spread_rate * 20))

        # Constante de tiempo variable: disminuye con el tiempo
        tau = 35 * (1 - 0.7 * min(1, (t - 20) / 40))
        exp_term = np.exp(-(t - 20) / tau)
        T = T_amb + (T20 - T_amb) * exp_term

        # Derivada: dT/dt = -(T20 - T_amb)/tau * exp_term * [1 + (t-20)/tau * 0.7/40]
        dTdt = -(T20 - T_amb) / tau * exp_term * (1 + (t - 20) / tau * 0.7 / 40)

    return T, dTdt

def get_celulosa_model_dadt():
    """
        Genera la ODE dα/dt para celulosa con T = T(t) tomada de la función global
        `funcion_temperatura(t)` (retorna T y dT/dt).

        Returns:
            tuple[callable, float]: (dadt_total, alpha0)
                - dadt_total(t, α): velocidad de conversión dα/dt
                - alpha0 (float): condición inicial (α=0)

        Notas:
            - Fase de humedad lineal hasta 100 °C (373 K) limitada por α_moisture.
            - Dos mecanismos separados por T_switch (350 °C).
            - Cambia la curva T(t) editando la variable global `funcion_temperatura`.
    """
    alpha0 = 0.0
    alpha_moisture = 0.25
    T_moisture_end = 100 + 273
    A1, Ea1 = 3e11, 138e3
    A2, Ea2 = 2e15, 180e3
    T_switch = 350 + 273

    def dadt_total(t, alpha):
        T ,dTdt = funcion_temperatura(t)
        if T <= T_moisture_end and alpha < alpha_moisture:
            return (alpha_moisture / T_moisture_end) * dTdt
        elif T <= T_switch:
            return A1 * np.exp(-Ea1/(R*T)) * ng15_f(alpha)
        else:
            return A2 * np.exp(-Ea2/(R*T)) * cr4_f(alpha)

    return dadt_total, alpha0

def get_xilano_model_dadt():
    """
        Genera la ODE dα/dt para xilano con T = T(t) tomada de la función global
        `funcion_temperatura(t)`.

        Returns:
            tuple[callable, float]: (dadt_total, alpha0)
    """
    alpha0 = 0.0
    alpha_moisture = 0.30
    T_moisture_end = 100 + 273
    A1, Ea1 = 9.8e13, 148e3
    A2, Ea2 = 2e2, 40e3
    T_switch = 285 + 273

    def dadt_total(t, alpha):
        T, dTdt = funcion_temperatura(t)
        if alpha > 0.95:
            return 0.0
        if T <= T_moisture_end and alpha < alpha_moisture:
            return (alpha_moisture / T_moisture_end) * dTdt
        elif T <= T_switch:
            return A1 * np.exp(-Ea1/(R*T)) * cr15_f(alpha)
        else:
            return A2 * np.exp(-Ea2/(R*T)) * cr15_f(alpha)

    return dadt_total, alpha0

def get_lignina_model_dadt( ):
    """
        Genera la ODE dα/dt para lignina con T = T(t) tomada de la función global
        `funcion_temperatura(t)`.

        Returns:
            tuple[callable, float]: (dadt_total, alpha0)
    """
    alpha0 = 0.0
    alpha_moisture = 0.40
    T_moisture_end = 100 + 273
    A1, Ea1 = 3e1, 38e3

    def dadt_total(t, alpha):
        T, dTdt = funcion_temperatura(t)
        if alpha > 0.96:
            return 0.0
        if T <= T_moisture_end and alpha < alpha_moisture:
            return (alpha_moisture / T_moisture_end) * dTdt
        else:
            return A1 * np.exp(-Ea1/(R*T)) * dm1_f(alpha)

    return dadt_total, alpha0

def simulacion_incendio():
    """
    Simula la degradación térmica global (mezcla C/X/L) bajo el perfil de temperatura
    T(t) seleccionado en la variable global `funcion_temperatura`, y muestra una figura
    con:
        (1) T(t) en °C
        (2) α_total(t)
        (3) dα/dt_total(t)

    Importante:
        - Esta función **no devuelve ningún valor**; únicamente genera y muestra la figura.
        - Para cambiar la curva T(t), modifica fuera de esta función la variable global:
              funcion_temperatura = iso834_modificada
              funcion_temperatura = curva_logaritmica
              funcion_temperatura = curva_exponencial
        - Pesos por defecto: [0.493, 0.227, 0.280] para (Celulosa, Xilano, Lignina).
        - t_eval = 35 000 puntos en 60 min para buena resolución temporal.

    Requisitos:
        - np, plt, solve_ivp
        - Constantes/funciones globales: R, ng15_f, cr4_f, cr15_f, dm1_f,
          y las funciones get_*_model_dadt() que leen `funcion_temperatura`.
    """
    # Parámetros mezcla y malla temporal
    weights = [0.493, 0.227, 0.28]  # C, X, L : Muestra 5 por defecto
    t_max = 60  # min
    t_eval = np.linspace(0, t_max, 35000)

    # Nombre legible de la curva para el título
    nombre_curva_map = {
        'iso834_modificada': 'ISO 834 modificada',
        'curva_logaritmica': 'Logarítmica',
        'curva_exponencial': 'Exponencial'
    }

    nombre_func = getattr(funcion_temperatura, '__name__', 'curva_personalizada')
    nombre_curva_legible = nombre_curva_map.get(nombre_func, 'Curva personalizada')

    # Precalcular T y dT/dt (con la curva elegida; aquí: ISO 834 modificada)
    T_values = np.zeros_like(t_eval)
    dTdt_values = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        T_values[i], dTdt_values[i] = funcion_temperatura(t) # Modificar aqui (tambien se puede automatizar) - Seleccionar modelo deseado

    # Resolver componentes (dα/dt) usando la misma curva global
    models = [get_celulosa_model_dadt(), get_xilano_model_dadt(), get_lignina_model_dadt()] # Modificar aqui tambien la funcion de temperatura
    alpha_components, dadt_components = [], []

    for model in models:
        dadt, alpha0 = model
        sol = solve_ivp(
            fun=lambda t, a: dadt(t, a[0]),
            t_span=(0, t_max),
            y0=[alpha0],
            t_eval=t_eval,
            method='RK45'
        )
        alpha_comp = sol.y[0]
        alpha_components.append(alpha_comp)

        # dα/dt del componente a lo largo de t_eval (misma curva T(t))
        dadt_comp = np.array([dadt(t, a) for t, a in zip(t_eval, alpha_comp)])
        dadt_components.append(dadt_comp)

    # Mezcla total
    alpha_total = sum(w * comp for w, comp in zip(weights, alpha_components))
    dadt_total = sum(w * comp for w, comp in zip(weights, dadt_components))

    # ---------- Figura ----------
    plt.figure(figsize=(7, 6))

    # 1) T(t)
    plt.subplot(3, 1, 1)
    plt.plot(t_eval, T_values - 273, 'r-', linewidth=2)
    plt.ylabel('Temperatura (°C)', fontsize = 10)
    plt.title('Perfil de Temperatura', fontsize = 12)

    # 2) α_total(t)
    plt.subplot(3, 1, 2)
    plt.plot(t_eval, alpha_total, 'b-', linewidth=2)
    plt.ylabel('Conversión α', fontsize = 10)
    plt.title('Degradación Térmica de la Muestra', fontsize = 12)

    # 3) dα/dt_total(t)
    plt.subplot(3, 1, 3)
    plt.plot(t_eval, dadt_total, 'g-', linewidth=2 )
    plt.ylabel('dα/dt', fontsize = 10)
    plt.xlabel('Tiempo (min)', fontsize = 10)
    plt.title('Velocidad de Degradación', fontsize = 12)
    plt.legend()

    # Suptítulo indicando la curva usada
    plt.suptitle(f"Simulación de Incendio — Curva T(t): {nombre_curva_legible}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Guardado en Resultados/SimulacionIncendio/<Nombre de la curva> ---
    out_dir = os.path.join('Resultados', 'F2.3_SimulacionIncendio', nombre_curva_legible)
    os.makedirs(out_dir, exist_ok=True)
    # nombre de archivo usando el nombre técnico de la función para evitar espacios raros
    archivo_png = f"Simulacion_Incendio_{nombre_func}.png"
    ruta_out = os.path.join(out_dir, archivo_png)
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figura guardada en: {ruta_out}")

# ========================= FASE 3.0 · ANÁLISIS DESACOPLADO =========================
def analisis_desacoplado(
    nombre_muestra: str = "muestra5",
    carpeta_vel: str = "VelocidadCalentamiento15",
    # Propiedades térmicas/geométricas (volumétricas)
    rho: float = 50.0,         # [kg/m3] densidad aparente del lecho
    Cp: float = 2000.0,        # [J/(kg·K)] calor específico efectivo
    h_loss: float = 2000.0,    # [W/(m3·K)] pérdidas convectivas por unidad de volumen
    T_inf: float = 300.0,      # [K] temperatura ambiente
    # Radiación (desactivada por defecto vía 'cr=0.0')
    epsilon: float = 0.90,
    sigma_sb: float = 5.670374419e-8,
    S_over_V: float = 200.0,
    cr: float = 0.0,           # factor de activación de radiación (0 ⇒ ignora radiación)
    # Precalentamiento con histéresis tipo “latch”
    h_heat: float = 2500.0,    # [W/(m3·K)]
    T_thr: float = 800.0,      # [K] umbral de desactivación del precalentamiento
    # Condiciones iniciales
    T0: float = 300.0,         # [K]
    alpha0_total: float = 0.01,# α_total inicial (se reparte según pesos)
    # Composición por defecto (Muestra 5)
    pesos: tuple = (0.493, 0.227, 0.28),  # (Celulosa, Xilano, Lignina) - Muestra 5
    # Parámetros del modelo de flujo de calor (ajustan la forma de la DSC “modelo”)
    H1_total: float = -371e3,  # [J/kg] área gaussiana 1
    H2_total: float =  6221e3, # [J/kg] área gaussiana 2
    mu1_C: float = 100.0,      # [°C] centro gaussiana 1
    mu2_C: float = 420.0,      # [°C] centro gaussiana 2
    sigma1_K: float = 26.7,    # [K] anchura gaussiana 1
    sigma2_K: float = 56.0,    # [K] anchura gaussiana 2
    beta_K_per_min: float = 15.0, # [K/min] tasa usada para construir la DSC “modelo”
    # Integración temporal
    t_final_s: float = 600.0,  # [s]
    npts: int = 3601           # nº de puntos en la malla temporal
):
    """
    Ejecuta un **análisis desacoplado 0D** del balance de energía con un término
    de **flujo de calor dependiente solo de T** (DSC “modelo” a partir de dos
    gaussianas) y **evolución cinética C/X/L** (vía dα/dt = A·exp(-Ea/RT)·f(α)).
    Genera **una única figura** con 3 subgráficas:
        (izq)   T(t) en °C
        (centro) α_total(t)
        (dcha)  DSC modelo (W/g) vs T (°C) **comparada** con DSC experimental

    Parámetros clave:
        - `nombre_muestra`, `carpeta_vel`: de dónde se toma la DSC experimental
          dentro de `DATOS_PROCESADOS['DatosMuestras']`.
        - `pesos`: fracciones (C, X, L) de la mezcla.
        - `H*_total`, `mu*_C`, `sigma*_K`: dan forma a la DSC “modelo”.
        - `beta_K_per_min`: tasa (K/min) usada SOLO para transformar H_eff(T)→W/g.
          No impone un programa de calentamiento; el calentamiento emerge del balance.

    Efectos:
        - Guarda la figura en:
          `Resultados/F3.0_AnalisisDesacoplado/<carpeta_vel>_<nombre_muestra>.png`
        - No devuelve nada (solo imprime la ruta guardada).
    """

    # ----------------------- Datos experimentales (DSC) -----------------------
    try:
        datos = DATOS_PROCESADOS['DatosMuestras'][carpeta_vel][nombre_muestra]
        T_exp_C = np.asarray(datos['temperature'])           # [°C]
        Q_exp_Wg = np.asarray(datos['heat_flow_normalized']) # [W/g]
        exp_disponible = True
    except Exception as e:
        print(f"[Aviso] No se encontró DSC experimental para {carpeta_vel}/{nombre_muestra}: {e}")
        T_exp_C = np.array([])
        Q_exp_Wg = np.array([])
        exp_disponible = False

    # ----------------------- DSC “modelo” (H_eff → W/g) -----------------------
    beta_K_per_s = beta_K_per_min / 60.0

    def _heff_gaussian(T_K, H_total, mu_K, sigma_K):
        """Gaussiana de H_eff(T) con área H_total [J/kg]. Devuelve [J/(kg·K)]."""
        A = H_total / (sigma_K * np.sqrt(2.0*np.pi))
        return A * np.exp(-0.5 * ((T_K - mu_K)/sigma_K)**2)

    mu1_K = mu1_C + 273.15
    mu2_K = mu2_C + 273.15

    def H_eff_T(T_K):
        """H_eff total [J/(kg·K)] como suma de dos gaussianas."""
        return (_heff_gaussian(T_K, H1_total, mu1_K, sigma1_K) + _heff_gaussian(T_K, H2_total, mu2_K, sigma2_K))

    def Heat_Flow_W_per_g(T_K):
        """
        Convierte H_eff(T) a flujo de calor específico [W/g] usando la tasa
        beta_K_per_s. Fórmula: W/g = H_eff(T)[J/(kg·K)] · beta[K/s] / 1000.
        """
        return (H_eff_T(T_K) * beta_K_per_s) / 1000.0

    # ----------------------- Cinética (1/s) C/X/L (anidadas) ------------------
    # Nota: A originales en [1/min] ⇒ se convierten a [1/s] dividiendo entre 60.
    def g_celulosa(T_K, a):
        A1, Ea1 = 3e11/60.0, 138e3
        A2, Ea2 = 2e15/60.0, 180e3
        T_switch = 350.0 + 273.0
        if a >= 1.0:
            return 0.0
        if T_K <= T_switch:
            return A1 * np.exp(-Ea1/(R*T_K)) * ng15_f(a)
        else:
            return A2 * np.exp(-Ea2/(R*T_K)) * cr4_f(a)

    def g_xilano(T_K, a):
        A1, Ea1 = 9.8e13/60.0, 148e3
        A2, Ea2 = 2e2/60.0,   40e3
        T_switch = 285.0 + 273.0
        if a >= 1.0:
            return 0.0
        if T_K <= T_switch:
            return A1 * np.exp(-Ea1/(R*T_K)) * cr15_f(a)
        else:
            return A2 * np.exp(-Ea2/(R*T_K)) * cr15_f(a)

    def g_lignina(T_K, a):
        A1, Ea1 = 3e1/60.0, 38e3
        if a >= 1.0:
            return 0.0
        return A1 * np.exp(-Ea1/(R*T_K)) * dm1_f(a)

    w_cel, w_xil, w_lig = pesos

    # ----------------------- RHS del sistema 0D -------------------------------
    # Estado: y = [T, a_c, a_x, a_l]
    def make_rhs():
        preheat_active = True  # latch: se desactiva al cruzar T_thr una vez

        def rhs(t, y):
            nonlocal preheat_active
            T, a_c, a_x, a_l = y

            # Cinética (clamp para evitar singularidades numéricas)
            a_c = float(np.clip(a_c, 0.001, 1.0))
            a_x = float(np.clip(a_x, 0.001, 1.0))
            a_l = float(np.clip(a_l, 0.001, 1.0))

            gC = g_celulosa(T, a_c)
            gX = g_xilano(T, a_x)
            gL = g_lignina(T, a_l)

            # Término de calor “volumétrico” (W/m3) a partir de W/g:
            q_vol = Heat_Flow_W_per_g(T) * rho * 1000.0

            # Histéresis de precalentamiento
            if preheat_active and T >= T_thr:
                preheat_active = False
            q_pre = (h_heat * (T_thr - T)) if (preheat_active and T < T_thr) else 0.0

            # Pérdidas totales (conv.+rad). Radiación apagada si cr=0.
            U_rad = (S_over_V * epsilon * sigma_sb) * (T**2 + T_inf**2) * (T + T_inf)
            U_vol = h_loss + cr*U_rad

            # Balance de energía
            dTdt = (-U_vol*(T - T_inf) + q_vol + q_pre) / (rho * Cp)

            # Evolución de conversiones
            da_c = 0.0 if a_c >= 1.0 else gC
            da_x = 0.0 if a_x >= 1.0 else gX
            da_l = 0.0 if a_l >= 1.0 else gL

            return [dTdt, da_c, da_x, da_l]

        return rhs

    # ----------------------- Integración temporal -----------------------------
    t_span = (0.0, float(t_final_s))
    t_eval = np.linspace(t_span[0], t_span[1], int(npts))

    a0_c = alpha0_total * w_cel
    a0_x = alpha0_total * w_xil
    a0_l = alpha0_total * w_lig
    y0 = [float(T0), float(a0_c), float(a0_x), float(a0_l)]

    rhs = make_rhs()
    sol = solve_ivp(rhs, t_span, y0, method='RK45', t_eval=t_eval, vectorized=False)

    t_min = sol.t / 60.0
    T_vec_K = sol.y[0]
    T_vec_C = T_vec_K - 273.15
    a_tot = np.clip(sol.y[1] + sol.y[2] + sol.y[3], 0.0, 1.0)

    # DSC “modelo” evaluada sobre rango de temperaturas experimental
    Q_model_Wg = Heat_Flow_W_per_g(T_exp_C+273)

    # ----------------------- Figura (3 subgráficas) ---------------------------
    fig, (axT, axA, axQ) = plt.subplots(1, 3, figsize=(15, 4))

    # (izq) T(t)
    axT.plot(t_min, T_vec_C, '-', lw=1.8, label='Modelo')
    axT.set_xlabel('t (min)', fontsize=11)
    axT.set_ylabel('T (°C)', fontsize=11)
    axT.set_title('Temperatura vs Tiempo', fontsize=12)

    # (centro) α_total(t)
    axA.plot(t_min, a_tot, '-', lw=1.8, color='#003366')
    axA.set_xlabel('t (min)', fontsize=11)
    axA.set_ylabel('α', fontsize=11)
    axA.set_title('Conversión vs Tiempo', fontsize=12)
    axA.set_ylim(0, 1.05)

    # (dcha) DSC modelo vs experimental (ambas vs T)
    axQ.plot(T_exp_C, Q_model_Wg, '-', lw=1.8, label='DSC modelo')
    axQ.plot(T_exp_C, Q_exp_Wg, '--', lw=1.2, label='DSC experimental', alpha=0.9)
    axQ.set_xlabel('T (°C)', fontsize=11)
    axQ.set_ylabel('Heat Flow (W/g)', fontsize=11)
    axQ.set_title('DSC: Modelo vs Experimental', fontsize=12)
    axQ.legend()

    fig.suptitle(f"Fase 3.0 · Análisis Desacoplado — {nombre_muestra} @ {carpeta_vel}", fontsize=13, y=1.02)
    fig.tight_layout()

    # Guardado
    out_dir = os.path.join("Resultados", "F3.0_AnalisisDesacoplado")
    os.makedirs(out_dir, exist_ok=True)
    nombre_png = f"{carpeta_vel}_{nombre_muestra}.png"
    ruta_out = os.path.join(out_dir, nombre_png)
    fig.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[F3.0] Figura guardada en: {ruta_out}")



def prediccion_acoplada(
    muestra="muestra5",
    carpeta_beta="VelocidadCalentamiento15",
    pesos=(0.493, 0.227, 0.28),     # Muestra 5
    # --- Parámetros térmicos (volumétricos) y condiciones ---
    rho=500.0,              # kg/m3  (densidad aparente)
    C=2000.0,               # J/(kg·K)  (calor específico efectivo)
    T_inf=300.0,            # K      (ambiente)
    h_loss=2000.0,          # W/(m3·K)  (convección/pérdidas)
    epsilon=0.90,           # (-)  emisividad
    sigma_sb=5.670374419e-8,# W/(m2·K4)  Stefan–Boltzmann
    S_over_V=200.0,         # m^-1  (superficie/volumen efectiva)
    cr_rad=1.0,             # (-)   factor corrector radiación
    # --- Término de calor de reacción y “precalentamiento” opcional ---
    H_eff=5.85e6,           # J/kg   (entalpía efectiva de degradación)
    h_heat=2500.0,          # W/(m3·K) (solo si T < T_thr y no se ha cruzado aún)
    T_thr=800.0,            # K  umbral del latch de precalentamiento
    # --- Integración temporal ---
    metodo="RK45",         # integrador (stiff friendly: 'Radau')
    rtol=1e-6, atol=1e-9
):
    """
    Predicción ACOPLADA 0D (energía + cinética) para una muestra de biomasa (mezcla C/X/L).

    Integra:
        dT/dt = [ -U_vol(T)*(T - T_inf) + ρ*H_eff*G(α,T) + z(T) ] / (ρ*C)
        da_i/dt = g_i(α_i, T)   (i ∈ {celulosa,xilano,lignina})

    donde:
        G = w_cel*g_c + w_xil*g_x + w_lig*g_l
        U_vol = h_loss + cr_rad*(S/V)*ε*σ*(T^2+T_inf^2)*(T+T_inf)
        z(T) = h_heat*(T_thr - T) mientras T < T_thr y no se haya cruzado (latch)

    Efectos:
        - No retorna nada; guarda una figura con 3 subplots:
            (1) T(t)   (K)
            (2) α_total(t)

    Args principales:
        muestra (str): 'muestra1'..'muestra5'.
        carpeta_beta (str): p.ej. 'VelocidadCalentamiento15' (para localizar la DSC exp).
        pesos (tuple): fracciones másicas (w_cel, w_xil, w_lig) que suman ≈1.
        H_eff (float): entalpía efectiva de degradación (J/kg).
        método/rtol/atol: parámetros del integrador.

    Notas:
        - Las A se usan en s⁻¹ (se convierten internamente desde min⁻¹).
        - La figura se guarda en: Resultados/F3.1_PrediccionAcoplada/<carpeta_beta>/PrediccionAcoplada_<muestra>.png
    """
    # --------- Recuperar Datos experimental (tiempo en min y Q_norm en W/g) ----------
    try:
        datos = DATOS_PROCESADOS['DatosMuestras'][carpeta_beta][muestra]
        t_exp_min = np.array(datos['time'], dtype=float)
    except KeyError:
        print(f"[Predicción acoplada] No encuentro datos de {muestra} en {carpeta_beta}.")
        return

    # Reescalar para que el tiempo empiece en 0
    t0_min = float(t_exp_min[0])
    t_exp_min = t_exp_min - t0_min

    # Mallado temporal para la integración (igual recorrido que el experimento)
    t_span = (0.0, 20.0 * 60.0)   # s
    npts = max(1000, len(t_exp_min))              # resolución acorde a exp
    t_eval = np.linspace(t_span[0], t_span[1], npts)

    # --------- Cinética componente (g_i) en s^-1: A(min^-1)/60 ----------
    def g_celulosa(T, a):
        # NG1.5 / CR4 con T_switch (K)
        A1, Ea1 = 3e11/60.0, 138e3
        A2, Ea2 = 2e15/60.0, 180e3
        T_sw = 350.0 + 273.0
        if a >= 1.0: return 0.0
        if T <= T_sw:
            return A1*np.exp(-Ea1/(R*T))*ng15_f(a)
        else:
            return A2*np.exp(-Ea2/(R*T))*cr4_f(a)

    def g_xilano(T, a):
        A1, Ea1 = 9.8e13/60.0, 148e3
        A2, Ea2 = 2e2/60.0,   40e3
        T_sw = 285.0 + 273.0
        if a >= 1.0: return 0.0
        if T <= T_sw:
            return A1*np.exp(-Ea1/(R*T))*cr15_f(a)
        else:
            return A2*np.exp(-Ea2/(R*T))*cr15_f(a)

    def g_lignina(T, a):
        A1, Ea1 = 3e1/60.0, 38e3
        if a >= 1.0: return 0.0
        # clamp para evitar singularidad en DM1 (1/(2α)) cerca de 0
        a_safe = np.clip(a, 1e-6, 0.999999)
        return A1*np.exp(-Ea1/(R*T))*dm1_f(a_safe)

    w_cel, w_xil, w_lig = pesos

    # --------- RHS acoplado con latch de precalentamiento ----------
    preheat_active = True  # se apaga cuando T cruza T_thr

    def rhs(t, y):
        nonlocal preheat_active
        T, a_c, a_x, a_l = y

        # Cinética (s^-1)
        gc = g_celulosa(T, a_c)
        gx = g_xilano(T, a_x)
        gl = g_lignina(T, a_l)
        G  = w_cel*gc + w_xil*gx + w_lig*gl

        # Latch precalentamiento
        if preheat_active and T >= T_thr:
            preheat_active = False
        z = (h_heat * (T_thr - T)) if (preheat_active and T < T_thr) else 0.0

        # Pérdidas totales (W/m3/K) y balance de energía (W/m3)
        U_rad = cr_rad*(S_over_V * epsilon * sigma_sb) * (T**2 + T_inf**2) * (T + T_inf)
        U_vol = h_loss + U_rad
        dTdt  = (-U_vol*(T - T_inf) + (rho*H_eff*G) + z) / (rho*C)

        # Evolución de α_i
        da_c = 0.0 if a_c >= 1.0 else gc
        da_x = 0.0 if a_x >= 1.0 else gx
        da_l = 0.0 if a_l >= 1.0 else gl
        return [dTdt, da_c, da_x, da_l]

    # --------- Condición inicial ---------
    T0 = T_inf    # empezamos al ambiente
    alpha0 = 0.01 # evitar 0 exacto por DM1
    y0 = [T0, alpha0*w_cel, alpha0*w_xil, alpha0*w_lig]

    # --------- Integración ---------
    sol = solve_ivp(rhs, t_span, y0, method=metodo, t_eval=t_eval, rtol=rtol, atol=atol)
    t_min = sol.t / 60.0
    T_vec = sol.y[0]
    a_tot = np.clip(sol.y[1] + sol.y[2] + sol.y[3], 0.0, 1.0)


    # --------- Figura (2 paneles) y guardado ---------
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # (1) T(t)
    axs[0].plot(t_min, T_vec, '-')
    axs[0].set_xlabel('t (min)')
    axs[0].set_ylabel('T (K)')
    axs[0].set_title('Temperatura')

    # (2) α_total(t)
    axs[1].plot(t_min, a_tot, '-')
    axs[1].set_xlabel('t (min)')
    axs[1].set_ylabel('α_total (-)')
    axs[1].set_title('Conversión total')

    fig.suptitle(f'Fase 3.1 — Predicción acoplada | {muestra} ', y=1.03, fontsize=12)
    fig.tight_layout()

    out_dir = os.path.join("Resultados", "F3.1_PrediccionAcoplada")
    os.makedirs(out_dir, exist_ok=True)
    ruta_out = os.path.join(out_dir, f"PrediccionAcoplada_{muestra}.png")
    fig.savefig(ruta_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Predicción acoplada] Figura guardada en: {ruta_out}")


if __name__ == "__main__":

    # Tratamiento de datos provenientes de la fase 0 experimental
    tratamiento_datos_exp(['DatosComponentes','DatosMuestras'])

    ## Fase 1. Obtención del triplete cinético a partir de datos de componentes
    #fase1_isoconversional()

    #fase1_criado() # Para ejecutar este, necesitas correr fase1_isoconversional antes

    #fase1_cr()  # Tarda mucho tiempo en ejecutarse, debido a las regresiones.

    ## Fase 2: Reconstrucción de las muestras - Validación del triplete cinético

    # Fase 2.0: Reconstrucción componentes
    #reconstruccion_celulosa()
    #reconstruccion_xilano()
    #reconstruccion_lignina()

    # Fase 2.1: Reconstrucción muestras composición conocida
    #reconstruccion_muestras()

    # Fase 2.2: Simulación combustión muestra con distintas velocidades de calentamiento
    #simulacion_muestra_betas()

    # Fase 2.3: Simulación combustión muestra bajo ecolución de temperatura "real"
    # ---------------------------------------------------
    # Selector GLOBAL (fuera de funciones) de la curva T(t)
    #       funcion_temperatura = iso834_modificada
    #       funcion_temperatura = curva_logaritmica
    #       funcion_temperatura = curva_exponencial
    # ---------------------------------------------------
    funcion_temperatura = iso834_modificada
    #simulacion_incendio()


    ## Fase 3: Implementacion en modelo 0D de incendios forestales
    # Fase 3.0: Analisis desacoplado
    analisis_desacoplado() # para esta si es necesario correr tratamiento_datos_exp (por la curva DSC)

    # Fase 3.1: Predicción acoplada
    prediccion_acoplada()

