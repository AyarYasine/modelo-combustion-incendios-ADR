# Modelo de combustión (0D ADR) + Cinética lignocelulósica

Repositorio con el **TFG** y el **código** para caracterizar la cinética de celulosa, xilano y lignina (en aire)
y acoplarla a un submodelo **ADR** (advección–difusión–reacción) de incendios forestales (formulación 0D).

## Contenidos
- `docs/TFG_YasineElAyar_2025.pdf`: memoria del TFG.
- `code/KineticsParameters.py`: script principal (Fases 0–3: isoconversionales, Criado, Coats–Redfern, reconstrucciones, simulaciones y acoplamiento ADR).
- `data/`: carpeta para datos (**no se versionan los datos crudos**).
- `results/`: salidas (figuras, excels, etc.).
- `CITATION.cff`, `LICENSE`, `requirements.txt`, `.gitignore`.

## Requisitos
Python 3.10+ (recomendado). Instala dependencias:
```bash
pip install -r requirements.txt
```

## ⚠️ Seguridad (Plotly)
El script original inicia sesión en Plotly para publicar gráficas. **No subas credenciales** al repositorio.
- Copia `.env.example` a `.env` y rellena `PLOTLY_USERNAME` y `PLOTLY_API_KEY` si deseas publicar gráficas.
- O bien comenta las líneas de autenticación en `KineticsParameters.py` y usa guardado local.

## Estructura sugerida de datos
```
data/
  raw/
    DatosComponentes/        # CSV/Excel de TGA/DSC de componentes puros
    DatosMuestras/           # CSV/Excel de TGA/DSC de mezclas
  processed/
results/
```
El script crea subcarpetas dentro de `results/` como `Resultados/F1.0_FWOStarinkFriedman/`, `Resultados/ReconstruccionMuestras/`, etc.

## Uso rápido
Desde la raíz del repo:
```bash
python code/KineticsParameters.py
```
Por defecto el __main__ activa:
- `tratamiento_datos_exp()` para preparar datos.
- `analisis_desacoplado()` y `prediccion_acoplada()` del modelo 0D ADR.

Activa/desactiva bloques del `__main__` según quieras ejecutar Fase 1 (FWO/Starink/Friedman), Fase 1.1 (Criado),
Fase 1.2 (Coats–Redfern), Fase 2 (reconstrucciones) o Fase 3 (acoplamiento).

> Si te faltan datos de ejemplo, crea la estructura y coloca tus archivos experimentales en `data/raw/`.
> Los resultados se guardarán en `results/Resultados/...`

## Publicación en GitHub (paso a paso)
1. Crea un repositorio vacío en GitHub (sin *README* ni *LICENSE*).
2. En tu máquina:
   ```bash
   git init
   git add .
   git commit -m "Primer commit: TFG + código"
   git branch -M main
   git remote add origin https://github.com/USUARIO/modelo-combustion-incendios-ADR.git
   git push -u origin main
   ```

## Licencia
Código bajo **MIT**. La memoria en `docs/` se distribuye con fines académicos; si necesitas otra licencia para el PDF,
indícalo en el README.

## Autoría
- **Yasine el Ayar** (2025).
