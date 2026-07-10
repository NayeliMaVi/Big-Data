# Runbook

## Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/run_pipeline.py
python -m streamlit run dashboard/app.py
```

El pipeline valida y prepara los artefactos finales desde el catálogo procesado de 3034 repositorios. No ejecuta `src/ingest.py` porque ese script genera un sample pequeño.

## Advertencia

No ejecutar `src/ingest.py` para regenerar el dataset final de 3034 repositorios, porque ese script genera un sample pequeño y puede sobrescribir `data/processed/repos_clean.csv`.
