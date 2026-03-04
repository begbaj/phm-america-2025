from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import signal
import json
from typing import Dict, Any

app = FastAPI()
templates = Jinja2Templates(directory="web_gui/templates")

TASKS_DIR = "tasks"
CONFIGS_DIR = os.path.join("web_gui", "task_configs")
USER_CONFIGS_DIR = os.path.join("web_gui", "user_configs")
OUTPUT_PLOTS_DIR = "img" 

# Ensure user configs dir exists
os.makedirs(USER_CONFIGS_DIR, exist_ok=True)

# Dictionary to track running processes: task_name -> process instance
running_tasks = {}

# Mount static files
app.mount("/img", StaticFiles(directory=OUTPUT_PLOTS_DIR), name="img")
app.mount("/static", StaticFiles(directory="web_gui/static"), name="static")

class SaveConfigModel(BaseModel):
    config_name: str
    data: Dict[str, Any]

@app.get("/user_configs/{task_name}", response_class=JSONResponse)
async def list_user_configs(task_name: str):
    """List all saved configurations for a specific task."""
    task_config_dir = os.path.join(USER_CONFIGS_DIR, task_name)
    if not os.path.exists(task_config_dir):
        return {"configs": []}
    
    configs = []
    for f in sorted(os.listdir(task_config_dir)):
        if f.endswith(".json"):
            configs.append(f.replace(".json", ""))
    return {"configs": configs}

@app.get("/user_configs/{task_name}/{config_name}", response_class=JSONResponse)
async def load_user_config(task_name: str, config_name: str):
    """Load a specific configuration."""
    file_path = os.path.join(USER_CONFIGS_DIR, task_name, f"{config_name}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Config not found")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user_configs/{task_name}", response_class=JSONResponse)
async def save_user_config(task_name: str, config: SaveConfigModel):
    """Save a configuration."""
    task_config_dir = os.path.join(USER_CONFIGS_DIR, task_name)
    os.makedirs(task_config_dir, exist_ok=True)
    
    file_path = os.path.join(task_config_dir, f"{config.config_name}.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(config.data, f, indent=4)
        return {"status": "success", "message": f"Configuration '{config.config_name}' saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    tasks = []
    if os.path.exists(TASKS_DIR):
        files = sorted([f for f in os.listdir(TASKS_DIR) if f.endswith(".py") and f != "__init__.py"])
        
        for filename in files:
            task_info = {"name": filename, "path": os.path.join(TASKS_DIR, filename), "config": None}
            
            json_name = filename.replace(".py", ".json")
            config_path = os.path.join(CONFIGS_DIR, json_name)
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        task_info["config"] = json.load(f)
                except Exception as e:
                    print(f"Error loading config for {filename}: {e}")
            
            tasks.append(task_info)
            
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
        relative_path = os.path.relpath(latest_file, OUTPUT_PLOTS_DIR)
        relative_path = relative_path.replace(os.path.sep, '/')
        return {"url": f"/img/{relative_path}", "filename": os.path.basename(latest_file), "timestamp": latest_time}
    return {"url": None}

@app.get("/browse_plots", response_class=JSONResponse)
async def browse_plots(path: str = ""):
    if path == "/":
        path = ""
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
        return JSONResponse({"status": "error", "message": f"Task {task_name} is already running."}, status_code=409)

    task_path = os.path.join(TASKS_DIR, task_name)
    if not os.path.exists(task_path):
        return JSONResponse({"status": "error", "message": f"Task {task_name} not found."}, status_code=404)
    
    try:
        body = await request.json()
    except Exception:
        body = {}

    cmd = ["python", "-u", task_path] # -u for unbuffered output is crucial!
    
    # Load config to check param types
    config = {}
    config_path = os.path.join(CONFIGS_DIR, task_name.replace(".py", ".json"))
    param_defs = {}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "groups" in config:
                    for group in config["groups"]:
                        for param in group.get("params", []):
                            param_defs[param["id"]] = param
        except Exception:
            pass

    for key, value in body.items():
        # Special handling for grouped preprocessing steps to map them back to individual flags
        if key == "preprocessing_steps" and isinstance(value, list):
            all_options = ["remove_outliers", "fill_missing"]
            for opt in all_options:
                flag_name = f"--{opt.replace('_', '-')}"
                flag_value = "True" if opt in value else "False"
                cmd.append(flag_name)
                cmd.append(flag_value)
            continue

        arg_name = f"--{key.replace('_', '-')}"
        
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        elif isinstance(value, list):
            fmt = "csv"
            if key in param_defs and "format" in param_defs[key]:
                fmt = param_defs[key]["format"]
            
            if fmt == "list":
                cmd.append(arg_name)
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(arg_name)
                cmd.append(",".join(str(v) for v in value))
        else:
            cmd.append(arg_name)
            cmd.append(str(value))

    async def event_generator():
        try:
            # Merge stdout and stderr for simpler streaming
            process = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.STDOUT
            )
            running_tasks[task_name] = process
            
            # Notify start
            yield json.dumps({"type": "status", "status": "running", "message": "Task started..."}) + "\n"

            # Stream output
            async for line in process.stdout:
                if line:
                    decoded = line.decode()
                    yield json.dumps({"type": "log", "data": decoded}) + "\n"
            
            await process.wait()
            
            if task_name in running_tasks:
                del running_tasks[task_name]

            if process.returncode == 0:
                yield json.dumps({"type": "status", "status": "success", "message": "Task completed successfully."}) + "\n"
            else:
                yield json.dumps({"type": "status", "status": "error", "message": f"Task failed with exit code {process.returncode}."}) + "\n"
                
        except asyncio.CancelledError:
            yield json.dumps({"type": "status", "status": "error", "message": "Task cancelled."}) + "\n"
        except Exception as e:
            if task_name in running_tasks:
                del running_tasks[task_name]
            yield json.dumps({"type": "status", "status": "error", "message": f"Execution error: {str(e)}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/stop_task/{task_name}")
async def stop_task(task_name: str):
    if task_name in running_tasks:
        process = running_tasks[task_name]
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            
            if task_name in running_tasks:
                del running_tasks[task_name]
                
            return {"status": "success", "message": f"Task {task_name} stopped successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to stop task: {str(e)}"}
    else:
        return {"status": "error", "message": f"Task {task_name} is not running."}