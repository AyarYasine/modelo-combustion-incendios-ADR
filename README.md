# Modelo de combustión (0D ADR) + Cinética lignocelulósica — **TFG**

Este repositorio contiene la **memoria del TFG** y el **código** para:
- procesar datos TGA/DSC de **celulosa**, **xilano** y **lignina** en **aire** y **N₂**,
- obtener parámetros cinéticos (FWO/Starink/Friedman, Criado, Coats–Redfern),
- **reconstruir** muestras y **simular** escenarios,
- y acoplar la cinética C/X/L a un **modelo 0D ADR** (balance de energía) para incendios.

> Código principal: `code/KineticsParameters.py`. Revisa/edita el bloque `if __name__ == "__main__":` para activar las fases que quieras ejecutar.

---

## 1) Requisitos

- **Python 3.10+** (recomendado 3.11)
- Paquetes: ver `requirements.txt`  
  Instala con:
  ```bash
  python -m venv .venv
  # macOS/Linux
  source .venv/bin/activate
  # Windows (PowerShell): .\.venv\Scripts\Activate.ps1

  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

### ⚠️ Nota sobre Plotly (Chart Studio)
El script incluye la subida de figuras interactivas (fase Criado) a **Plotly Chart Studio**. **No publiques tus credenciales** en el código. Usa variables de entorno:
```python
# Sustituye el inicio de sesión duro por:
import os, chart_studio.plotly as py
py.sign_in(os.getenv("PLOTLY_USERNAME"), os.getenv("PLOTLY_API_KEY"))
```
Crea un archivo `.env` (no se versiona) a partir de `.env.example` con tus claves.
Si no quieres usar Plotly online, comenta las líneas de `sign_in` y deja solo el guardado local de HTML.

---

## 2) Estructura de carpetas (datos y resultados)

Para que se puedar ejecute el código **sin preparar nada**, este repo **incluye**:
- `DatosComponentes/` — datos experimentales de componentes puros
- `DatosMuestras/` — datos experimentales de muestras (muestra1..muestra5)

Asimismo, se incluyen los resultados esperados al ejecutar las distintas fases del código:
- `Resultados/` — figuras y Excels **generados por cada fase**

### Distribución esperada de datos
```
DatosComponentes/
  VelocidadCalentamiento5/
  VelocidadCalentamiento15/
  VelocidadCalentamiento30/
DatosMuestras/
  VelocidadCalentamiento5/
  VelocidadCalentamiento15/
  VelocidadCalentamiento30/
Resultados/            # (se completa automáticamente al ejecutar)
```

**Formato de los CSV** (separador `;`, decimales con `,`):  
Columnas obligatorias (encabezados exactos):
```
Time t (min)
Temperature T(c)
Weight (mg)
Weight (%)
Heat Flow Q (mW)
Heat Flow (Normalized) Q (W/g)
```
> El cargador convierte comas → puntos y castea a `float` automáticamente.

**Convención de nombres (recomendada para Componentes):** `{componente}_{beta}_{atm}.csv`, p. ej.:
```
celulosa_15_aire.csv
xilano_5_n2.csv
lignina_30_aire.csv
```
En `DatosMuestras/VelocidadCalentamientoXX/` usa `muestra1.csv`, ..., `muestra5.csv` (o nombres coherentes con lo que actives).

---

## 3) Ejecución rápida (con datos ya incluidos)

Desde la **raíz** del repo:
```bash
python code/KineticsParameters.py
```
Por defecto se ejecuta la fase inicial de tratamiento de datos experimentales en el bloque `__main__`:
- `tratamiento_datos_exp(['DatosComponentes','DatosMuestras'])`

Para activar/desactivar fases, abre `code/KineticsParameters.py` y comenta/descomenta llamadas en el `__main__`.

---

## 4) Fases del flujo y salidas generadas

> Las rutas de salida se crean dentro de `Resultados/…` automáticamente.

### **Fase 0 — Tratamiento de datos experimentales**
Función: `tratamiento_datos_exp(['DatosComponentes','DatosMuestras'])`  
- Lee todos los CSV de `VelocidadCalentamiento{5,15,30}`.
- Calcula α, dα/dT (y dα/dt multiplicando por β) y **DTG** suavizadas.
- Grafica multieje **α / dα/dt / Heat Flow (W/g)** con particiones por temperaturas seleccionadas.
- Integra **DSC** en dos tramos (≈ 200 °C) y exporta a Excel.

**Salidas:**
```
Resultados/F0_GraficasComponentes/VelocidadCalentamientoX/<nombre>.png
Resultados/F0_GraficasMuestras/VelocidadCalentamientoX/<nombre>.png
Resultados/F0_CurvaDSC/Componentes|Biomasa/VelocidadCalentamientoX/<nombre>.xlsx
```

---

### **Fase 1.0 — Isoconversional (FWO, Starink, Friedman)**
Funciones: `aplicar_fwo_starink_friedman()` y `fase1_isoconversional()`  
- Requiere **3 betas** (5, 15, 30 °C/min) por componente/atmósfera.
- Ajusta **ln(β)**, **ln(β/T^1.92)** y **ln(dα/dt)** vs **1/T** por α ∈ [0.10, 0.90].
- Devuelve **Ea(α)** y **R²** para cada método.  
**Salidas:**
```
Resultados/F1.0_FWOStarinkFriedman/<Atmósfera>/<Componente>/
  FWO_Starink_Friedman_<componente>_<atmósfera>.png|xlsx
```

---

### **Fase 1.1 — Criado (Master Plots)**
Funciones: `calcular_z_alpha_criado_exp()`, `calcular_master_plots_teoricos()`, `fase1_criado()`  
- Usa resultados de Fase 1.0 + series de Friedman.
- Construye Z(α) **experimental** y lo compara con familias **teóricas** (NG, DM, CR, etc.).
- Guarda **HTML interactivo** y sube (opcional) a Plotly Chart Studio.
**Salidas:**
```
Resultados/F1.1_Criado/<Atmósfera>/<Componente>/
  Criado_<comp>_<beta>_<atm>.html
  Criado_<comp>_<beta>_<atm>_url.txt   # enlace Chart Studio (si activado)
```

---

### **Fase 1.2 — Coats–Redfern (integral)**
Funciones: `ajustar_coats_redfern()`, `aplicar_modelos_cr()`, `fase1_cr()`  
- Divide los datos por **subgrupos de T** (según temperaturas seleccionadas).
- Ajusta todos los **g(α)** clásicos (CR, DM, NG, …) por subgrupo.
- Exporta **Ea**, **A**, **R²** y genera **gráficas de regresión**.
**Salidas:**
```
Resultados/F1.2_CoatsRedfern/<VelocidadCalentamientoX>/<nombre>.xlsx
Resultados/F1.2_RegresionesCR/<VelocidadCalentamientoX>/<nombre>_sub<i>_parte<j>.png
```
> **Aviso:** esta fase puede tardar por el número de regresiones.

---

### **Fase 2.0 — Reconstrucción de componentes puros**
Funciones: `reconstruccion_celulosa()`, `reconstruccion_xilano()`, `reconstruccion_lignina()`  
- Resuelve ODE **dα/dT** por componente (humedad + mecanismos por T).
- Compara **α(T)** y **dα/dT(T)** con el experimento (β=15 por defecto).
**Salidas:**
```
Resultados/F2.0_ReconstruccionComponentes/<Componente>/Reconstruccion_*.png
```

---

### **Fase 2.1 — Reconstrucción de muestras (composición conocida)**
Funciones: `reconstruccion_muestras()`, `reconstruir_muestra_unica()`  
- Combina α(T) de C/X/L con **pesos** (p. ej. Muestra5: 0.493/0.227/0.280).
- Gráfica con **α(T)** (eje izq.) y **dα/dT(T)** (eje der.).
**Salidas:**
```
Resultados/F2.1_ReconstruccionMuestras/MuestraX/Reconstruccion_*.png
```

---

### **Fase 2.2 — Efecto de β en muestra 5**
Función: `simulacion_muestra_betas()`  
- Integra ODEs para β ∈ {5,10,15,20,30} y compara curvas.
**Salidas:**
```
Resultados/F2.2_SimulacionMuestraBetas/Simulacion_Muestra5_multibeta.png
```

---

### **Fase 2.3 — Incendio con T(t) “realista”**
Funciones: `iso834_modificada()`, `curva_exponencial()`, `curva_logaritmica()`, `simulacion_incendio()`  
- Selecciona la curva global en el `__main__`:
  ```python
  funcion_temperatura = iso834_modificada  # o curva_logaritmica, curva_exponencial
  # simulacion_incendio()   # descomenta para ejecutar
  ```
**Salidas:**
```
Resultados/F2.3_SimulacionIncendio/<NombreCurva>/Simulacion_Incendio_<func>.png
```

---

### **Fase 3.0 — Análisis desacoplado (0D)**
Función: `analisis_desacoplado()`  
- Balance de energía con **DSC “modelo”** (dos gaussianas) + cinética C/X/L.
- Compara **DSC modelo** vs **DSC experimental** de la muestra.
**Salidas:**
```
Resultados/F3.0_AnalisisDesacoplado/<Velocidad>_<Muestra>.png
```

---

### **Fase 3.1 — Predicción acoplada (0D)**
Función: `prediccion_acoplada()`  
- Integra **energía + cinética** con entalpía efectiva `H_eff` y latch de precalentamiento.
- Devuelve **T(t)** y **α_total(t)** acoplados.
**Salidas:**
```
Resultados/F3.1_PrediccionAcoplada/PrediccionAcoplada_<muestra>.png
```

---

## 5) Cómo reproducir **exactamente** lo del TFG (con este repo)

1. Clona y crea entorno + deps (`pip install -r requirements.txt`).  
2. Verifica que están presentes `DatosComponentes/` y `DatosMuestras/` con **las 3 betas** por componente/atmósfera.  
3. Ejecuta:
   ```bash
   python code/KineticsParameters.py
   ```
4. Revisa `Resultados/` y compáralo con lo ya incluido (deberían coincidir salvo pequeñas diferencias numéricas).

---


## 6) Problemas habituales

- **“No encuentra datos”** → verifica rutas: `DatosComponentes/VelocidadCalentamiento{5,15,30}/` y nombres `celulosa_XX_atm.csv`.  
- **“Se requieren 3 velocidades”** (Fase 1.0) → añade 5, 15 y 30 para ese componente/atmósfera.  
- **CSV con punto decimal** → convierte a **coma** o adapta el loader.  
- **`chart_studio` no instalado** → `pip install chart-studio` (ya en `requirements.txt`).  
- **Archivos muy grandes** → considera Git LFS o subir un subconjunto representativo + enlace externo.

---

## 7) Licencia
- Código: **MIT** (ver `LICENSE`).
- Documentos/figuras: uso académico. Atribuye cuando corresponda.
