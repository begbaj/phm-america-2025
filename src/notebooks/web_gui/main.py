from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import subprocess
import os

app = FastAPI()
templates = Jinja2Templates(directory="web_gui/templates")

TASKS_DIR = "tasks"
OUTPUT_PLOTS_DIR = "output_plots" # Cartella dove vengono salvati i grafici

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    tasks = []
    for filename in sorted(os.listdir(TASKS_DIR)):
        if filename.endswith(".py"):
            tasks.append({"name": filename, "path": os.path.join(TASKS_DIR, filename)})
    return templates.TemplateResponse("index.html", {"request": request, "tasks": tasks})

@app.post("/run_task/{task_name}")
async def run_task(task_name: str, request: Request):
    task_path = os.path.join(TASKS_DIR, task_name)
    if not os.path.exists(task_path):
        return {"status": "error", "message": f"Task {task_name} not found."}
    
    # Leggi eventuale JSON body con parametri (es. {"target": "HPC"})
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Costruisci il comando da eseguire (passa --target se fornito)
    cmd = ["python", task_path]
    if isinstance(body, dict):
        if "target" in body and body["target"]:
            cmd += ["--target", str(body["target"]) ]
        # Se sono stati selezionati modelli (es. ["linear","rf"]) li passiamo come --models linear,rf
        if "models" in body and isinstance(body["models"], list) and body["models"]:
            cmd += ["--models", ",".join(body["models"]) ]
        # PREPROCESSING specific options
        if "steps" in body and isinstance(body["steps"], list) and body["steps"]:
            cmd += ["--steps", ",".join(body["steps"]) ]
        # FEATURE ENGINEERING specific options
        if "statistical_features" in body and body["statistical_features"]:
            cmd += ["--statistical-features", str(body["statistical_features"]) ]
        if "pipeline_window" in body and body["pipeline_window"]:
            cmd += ["--pipeline-window", str(body["pipeline_window"]) ]
        if "pipeline_step" in body and body["pipeline_step"]:
            cmd += ["--pipeline-step", str(body["pipeline_step"]) ]
        # Preprocessing method options
        if "outlier_method" in body and body["outlier_method"]:
            cmd += ["--outlier-method", str(body["outlier_method"]) ]
        if "outlier_threshold" in body and body["outlier_threshold"]:
            cmd += ["--outlier-threshold", str(body["outlier_threshold"]) ]
        if "smoothing_window" in body and body["smoothing_window"]:
            cmd += ["--smoothing-window", str(body["smoothing_window"]) ]
        if "smoothing_step" in body and body["smoothing_step"]:
            cmd += ["--smoothing-step", str(body["smoothing_step"]) ]
        if "smoothing_method" in body and body["smoothing_method"]:
            cmd += ["--smoothing-method", str(body["smoothing_method"]) ]
        # RESIDUAL ANALYSIS specific options
        if "target_rul" in body and body["target_rul"]:
            cmd += ["--target-rul", str(body["target_rul"]) ]
        if "healthy_window" in body and body["healthy_window"]:
            cmd += ["--healthy-window", str(body["healthy_window"]) ]
        # MODEL TRAINING specific options
        if "target_training" in body and body["target_training"]:
            cmd += ["--target-training", str(body["target_training"]) ]
        if "healthy_window" in body and body["healthy_window"]:
            cmd += ["--healthy-window", str(body["healthy_window"]) ]
        
        # Random Forest params
        if "rf_n_estimators" in body and body["rf_n_estimators"]:
            cmd += ["--rf-n-estimators", str(body["rf_n_estimators"]) ]
        if "rf_max_depth" in body and body["rf_max_depth"]:
            cmd += ["--rf-max-depth", str(body["rf_max_depth"]) ]
            
        # XGBoost params
        if "xgb_n_estimators" in body and body["xgb_n_estimators"]:
            cmd += ["--xgb-n-estimators", str(body["xgb_n_estimators"]) ]
        if "xgb_learning_rate" in body and body["xgb_learning_rate"]:
            cmd += ["--xgb-learning-rate", str(body["xgb_learning_rate"]) ]
        if "xgb_max_depth" in body and body["xgb_max_depth"]:
            cmd += ["--xgb-max-depth", str(body["xgb_max_depth"]) ]
            
        # Transformer params
        if "trans_epochs" in body and body["trans_epochs"]:
            cmd += ["--trans-epochs", str(body["trans_epochs"]) ]
        if "trans_batch_size" in body and body["trans_batch_size"]:
            cmd += ["--trans-batch-size", str(body["trans_batch_size"]) ]
        if "trans_learning_rate" in body and body["trans_learning_rate"]:
            cmd += ["--trans-learning-rate", str(body["trans_learning_rate"]) ]

    # Esegui il task in un sottoprocesso
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        return {"status": "success", "message": f"Task {task_name} completed successfully.", "stdout": stdout.decode(), "stderr": stderr.decode()}
    else:
        return {"status": "error", "message": f"Task {task_name} failed.", "stdout": stdout.decode(), "stderr": stderr.decode()} 

@app.get("/list_plots", response_class=JSONResponse)
async def list_plots():
    image_files = []
    if os.path.exists(OUTPUT_PLOTS_DIR):
        for root, _, files in os.walk(OUTPUT_PLOTS_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    # Create a URL path relative to the /static mount
                    relative_path = os.path.relpath(os.path.join(root, file), OUTPUT_PLOTS_DIR)
                    image_files.append(f"/static/{relative_path}")
    return {"plots": image_files}


# Per servire i file statici (es. i grafici)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=OUTPUT_PLOTS_DIR), name="static")
