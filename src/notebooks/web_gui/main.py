from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import os
import signal

app = FastAPI()
templates = Jinja2Templates(directory="web_gui/templates")

TASKS_DIR = "tasks"
OUTPUT_PLOTS_DIR = "img" # Cartella dove vengono salvati i grafici

# Dictionary to track running processes: task_name -> process instance
running_tasks = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    tasks = []
    for filename in sorted(os.listdir(TASKS_DIR)):
        if filename.endswith(".py"):
            tasks.append({"name": filename, "path": os.path.join(TASKS_DIR, filename)})
    return templates.TemplateResponse("index.html", {"request": request, "tasks": tasks})

@app.get("/latest_plot", response_class=JSONResponse)
async def get_latest_plot():
    latest_file = None
    latest_time = 0
    
    if os.path.exists(OUTPUT_PLOTS_DIR):
        for root, _, files in os.walk(OUTPUT_PLOTS_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    full_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_file = full_path
                    except OSError:
                        pass
                        
    if latest_file:
        # relative_path must be relative to the StaticFiles directory (OUTPUT_PLOTS_DIR)
        relative_path = os.path.relpath(latest_file, OUTPUT_PLOTS_DIR)
        # Ensure forward slashes for URL
        relative_path = relative_path.replace(os.path.sep, '/')
        return {"url": f"/static/{relative_path}", "filename": os.path.basename(latest_file), "timestamp": latest_time}
    return {"url": None}

@app.get("/browse_plots", response_class=JSONResponse)
async def browse_plots(path: str = ""):
    # Handle root path request
    if path == "/":
        path = ""
        
    # Prevent directory traversal
    if ".." in path:
        return {"error": "Invalid path"}

    full_path = os.path.join(OUTPUT_PLOTS_DIR, path)
    
    if not os.path.exists(full_path):
         return {"error": "Path not found"}
         
    if not os.path.isdir(full_path):
        return {"error": "Not a directory"}

    items = {"current_path": path, "dirs": [], "files": []}
    
    try:
        with os.scandir(full_path) as it:
            for entry in it:
                if entry.name.startswith('.'): continue
                if entry.is_dir():
                    items["dirs"].append(entry.name)
                elif entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    items["files"].append(entry.name)
    except Exception as e:
        return {"error": str(e)}
        
    items["dirs"].sort()
    items["files"].sort()
    return items

@app.post("/run_task/{task_name}")
async def run_task(task_name: str, request: Request):
    if task_name in running_tasks:
        return {"status": "error", "message": f"Task {task_name} is already running."}

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

    # Esegui il task in un sottoprocesso asincrono
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        running_tasks[task_name] = process
        
        stdout, stderr = await process.communicate()
        
        # Rimuovi dal dizionario quando completato
        if task_name in running_tasks:
            del running_tasks[task_name]

        if process.returncode == 0:
            return {"status": "success", "message": f"Task {task_name} completed successfully.", "stdout": stdout.decode(), "stderr": stderr.decode()}
        else:
            return {"status": "error", "message": f"Task {task_name} failed.", "stdout": stdout.decode(), "stderr": stderr.decode()}
    except asyncio.CancelledError:
        return {"status": "error", "message": f"Task {task_name} was cancelled."}
    except Exception as e:
        if task_name in running_tasks:
            del running_tasks[task_name]
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

@app.post("/stop_task/{task_name}")
async def stop_task(task_name: str):
    if task_name in running_tasks:
        process = running_tasks[task_name]
        try:
            process.terminate()
            # Attendi brevemente che il processo termini
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill() # Forza la chiusura se non risponde
            
            if task_name in running_tasks:
                del running_tasks[task_name]
                
            return {"status": "success", "message": f"Task {task_name} stopped successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to stop task: {str(e)}"}
    else:
        return {"status": "error", "message": f"Task {task_name} is not running."}

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
