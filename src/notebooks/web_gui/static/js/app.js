// ============== UTILITIES ==============
function toggleTask(header) {
    const taskItem = header.parentElement;
    const content = taskItem.querySelector('.task-content');
    const arrow = header.querySelector('.toggle-arrow');
    
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        arrow.style.transform = 'rotate(0deg)';
    } else {
        content.classList.add('hidden');
        arrow.style.transform = 'rotate(-90deg)';
    }
}

function saveFormState() {
    const formData = {};
    document.querySelectorAll('input, select').forEach(el => {
        // Skip form elements inside config modal
        if (el.closest('#config-modal')) return;
        
        if (el.type === 'checkbox') {
            formData[el.id] = el.checked;
        } else {
            formData[el.id] = el.value;
        }
    });
    localStorage.setItem('preprocessingFormState', JSON.stringify(formData));
}

function loadFormState() {
    const saved = localStorage.getItem('preprocessingFormState');
    if (saved) {
        try {
            const formData = JSON.parse(saved);
            Object.keys(formData).forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    if (el.type === 'checkbox') {
                        el.checked = formData[id];
                    } else {
                        el.value = formData[id];
                    }
                }
            });
        } catch (e) {
            console.error('Error loading form state:', e);
        }
    }
}

function resetForm(taskName) {
    const taskId = taskName.replace(/\./g, '_');
    const form = document.getElementById('form-' + taskId);
    
    if (form) {
        form.querySelectorAll('input, select').forEach(el => {
            if (el.type === 'checkbox') {
                el.checked = el.defaultChecked;
            } else if (el.dataset.default !== undefined) {
                el.value = el.dataset.default;
            } else {
                el.value = el.defaultValue;
            }
        });
        saveFormState();
        showToast(`Form reset for ${taskName}`, 'info');
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    
    const baseClasses = "mb-4 p-4 rounded-lg shadow-lg text-white transform transition-all duration-300 translate-x-0 flex items-center";
    let typeClasses = "bg-blue-500";
    if (type === 'success') typeClasses = "bg-green-500";
    if (type === 'error') typeClasses = "bg-red-500";
    
    toast.className = `${baseClasses} ${typeClasses}`;
    toast.innerHTML = `
        <span class="mr-2">${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
        <span>${message}</span>
    `;
    
    toast.style.transform = "translateX(100%)";
    container.appendChild(toast);
    
    requestAnimationFrame(() => {
        toast.style.transform = "translateX(0)";
    });

    setTimeout(() => {
        toast.style.transform = "translateX(100%)";
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function copyOutput() {
    const resultsDiv = document.getElementById('results-content');
    if (!resultsDiv) return;
    const text = resultsDiv.innerText;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Output copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy output', 'error');
    });
}

// ============== CONFIGURATION MANAGEMENT ==============
let currentConfigTask = null;

function openConfigModal(taskName, mode) {
    currentConfigTask = taskName;
    const modal = document.getElementById('config-modal');
    const title = document.getElementById('config-modal-title');
    const saveMode = document.getElementById('config-save-mode');
    const loadMode = document.getElementById('config-load-mode');
    
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    
    if (mode === 'save') {
        title.textContent = `Save Config: ${taskName}`;
        saveMode.classList.remove('hidden');
        loadMode.classList.add('hidden');
        document.getElementById('config-name-input').value = '';
        setTimeout(() => document.getElementById('config-name-input').focus(), 50);
    } else {
        title.textContent = `Load Config: ${taskName}`;
        saveMode.classList.add('hidden');
        loadMode.classList.remove('hidden');
        loadConfigsList(taskName);
    }
}

function closeConfigModal() {
    const modal = document.getElementById('config-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    currentConfigTask = null;
}

async function loadConfigsList(taskName) {
    const listContainer = document.getElementById('config-list');
    listContainer.innerHTML = '<div class="text-center text-gray-400 text-sm py-4"><div class="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-500 mx-auto mb-2"></div>Loading...</div>';
    
    try {
        const response = await fetch(`/user_configs/${taskName}`);
        const data = await response.json();
        
        listContainer.innerHTML = '';
        if (data.configs && data.configs.length > 0) {
            data.configs.forEach(configName => {
                const item = document.createElement('div');
                item.className = 'flex justify-between items-center p-3 hover:bg-gray-100 rounded-lg cursor-pointer group transition border border-transparent hover:border-gray-200';
                item.innerHTML = `
                    <span class="font-medium text-gray-700 flex items-center"><i class="fas fa-file-code mr-3 text-indigo-400"></i>${configName}</span>
                    <button class="text-indigo-600 opacity-0 group-hover:opacity-100 text-xs font-bold bg-indigo-50 px-3 py-1.5 rounded-full hover:bg-indigo-100 transition-opacity" onclick="loadConfig('${configName}')">Load</button>
                `;
                listContainer.appendChild(item);
            });
        } else {
            listContainer.innerHTML = '<div class="text-center text-gray-400 text-sm py-4">No saved configurations found.</div>';
        }
    } catch (e) {
        listContainer.innerHTML = '<div class="text-center text-red-400 text-sm py-4">Error loading configs.</div>';
    }
}

function getConfigFormData(taskId) {
    const form = document.getElementById('form-' + taskId);
    const formData = {};
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(el => {
        if (el.type === 'checkbox') {
            if (el.checked) {
                const name = el.name.replace('[]', '');
                if (el.name.includes('[]')) {
                    if (!formData[name]) formData[name] = [];
                    formData[name].push(el.value);
                } else {
                    formData[name] = true;
                }
            }
        } else {
            formData[el.name] = el.value;
        }
    });
    return formData;
}

async function confirmSaveConfig() {
    const nameInput = document.getElementById('config-name-input');
    const configName = nameInput.value.trim();
    if (!configName) {
        showToast('Please enter a configuration name', 'error');
        return;
    }
    
    const taskId = currentConfigTask.replace(/\./g, '_');
    const formData = getConfigFormData(taskId);
    
    try {
        const response = await fetch(`/user_configs/${currentConfigTask}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_name: configName, data: formData })
        });
        const res = await response.json();
        
        if (response.ok) {
            showToast(res.message, 'success');
            closeConfigModal();
        } else {
            showToast(res.detail || 'Error saving config', 'error');
        }
    } catch (e) {
        showToast('Network error saving config', 'error');
    }
}

async function loadConfig(configName) {
    try {
        const response = await fetch(`/user_configs/${currentConfigTask}/${configName}`);
        const data = await response.json();
        
        if (response.ok) {
            const taskId = currentConfigTask.replace(/\./g, '_');
            const form = document.getElementById('form-' + taskId);
            
            // Clear multiselects first (uncheck all)
            form.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);

            Object.keys(data).forEach(key => {
                const val = data[key];
                if (Array.isArray(val)) {
                    const checkboxes = form.querySelectorAll(`input[name="${key}[]"]`);
                    checkboxes.forEach(cb => {
                        if (val.includes(cb.value)) cb.checked = true;
                    });
                } else if (typeof val === 'boolean') {
                    const cb = form.querySelector(`input[name="${key}"]`);
                    if (cb) cb.checked = val;
                } else {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input) input.value = val;
                }
            });
            
            saveFormState(); // Persist loaded state
            showToast(`Configuration '${configName}' loaded`, 'success');
            closeConfigModal();
        } else {
            showToast('Error loading config', 'error');
        }
    } catch (e) {
        showToast('Network error loading config', 'error');
    }
}

// ============== MODAL & ZOOM & PAN LOGIC ==============
let currentZoom = 1;
let isDragging = false;
let startX = 0;
let startY = 0;
let translateX = 0;
let translateY = 0;

function openModal(imgSrc) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-img');
    img.src = imgSrc;
    
    // Reset view
    currentZoom = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
    
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeModal(event) {
    if (event) event.stopPropagation();
    const modal = document.getElementById('image-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

function zoomIn() {
    currentZoom += 0.2;
    updateTransform();
}

function zoomOut() {
    if (currentZoom > 0.4) {
        currentZoom -= 0.2;
        updateTransform();
    }
}

function updateTransform() {
    const img = document.getElementById('modal-img');
    if(img) {
        img.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentZoom})`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Config modal close logic
    document.getElementById('config-modal').addEventListener('click', (e) => {
        if(e.target.id === 'config-modal') closeConfigModal();
    });

    // Image Modal Interaction
    const modalImg = document.getElementById('modal-img');
    if(modalImg){
        // Wheel Zoom
        modalImg.addEventListener('wheel', function(e) {
            e.preventDefault();
            if (e.deltaY < 0) zoomIn();
            else zoomOut();
        });
        
        // Panning
        modalImg.addEventListener('mousedown', (e) => {
            isDragging = true;
            // Record start pos relative to current translate
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
            modalImg.style.cursor = 'grabbing';
            e.preventDefault(); 
        });
        
        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();
            translateX = e.clientX - startX;
            translateY = e.clientY - startY;
            updateTransform();
        });
        
        window.addEventListener('mouseup', () => {
            if(isDragging) {
                isDragging = false;
                modalImg.style.cursor = 'grab';
            }
        });
    }
    
    loadFormState();
    refreshPlots();
});

// ============== MAIN TASK EXECUTION ==============
async function runTask(taskName) {
    const taskId = taskName.replace(/\./g, '_');
    const runBtn = document.getElementById(`btn-run-${taskId}`);
    const stopBtn = document.getElementById(`btn-stop-${taskId}`);
    
    if (runBtn) runBtn.classList.add('hidden');
    if (stopBtn) stopBtn.classList.remove('hidden');
    
    const resultsContent = document.getElementById('results-content');
    
    // Reset console
    resultsContent.className = 'bg-gray-900 text-gray-200 p-4 rounded-lg overflow-x-auto font-mono text-sm max-h-[500px] overflow-y-auto shadow-inner border border-gray-800';
    resultsContent.innerHTML = `<div class="flex items-center space-x-2 py-2 border-b border-gray-700 mb-2"><div class="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-400"></div><span class="text-indigo-300 font-bold">Initializing ${taskName}...</span></div><div id="console-stream" class="whitespace-pre-wrap"></div>`;
    
    const consoleStream = document.getElementById('console-stream');
    const body = getConfigFormData(taskId);

    try {
        const response = await fetch(`/run_task/${taskName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep the last partial line

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    
                    if (msg.type === 'log') {
                        // Append log line
                        // Simple text append is faster than creating elements for high frequency logs
                        // Using insertAdjacentHTML for basic coloring if needed, but textContent is safer.
                        // We use escapeHtml to be safe but allow HTML structure if we wanted.
                        consoleStream.innerHTML += escapeHtml(msg.data); 
                        
                        // Auto-scroll to bottom
                        resultsContent.scrollTop = resultsContent.scrollHeight;
                        
                    } else if (msg.type === 'status') {
                        if (msg.status === 'running') {
                            // Update header maybe?
                        } else if (msg.status === 'success') {
                            showToast(msg.message, 'success');
                            // Refresh plots on success
                            refreshPlots();
                        } else if (msg.status === 'error') {
                            showToast(msg.message, 'error');
                            consoleStream.innerHTML += `\n<span class="text-red-500 font-bold">ERROR: ${escapeHtml(msg.message)}</span>`;
                        }
                    }
                } catch (e) {
                    console.error('Error parsing stream line:', line, e);
                }
            }
        }
        
    } catch (error) {
        consoleStream.innerHTML += `\n<div class="text-red-500 p-4">Network Error: ${error.message}</div>`;
        showToast('Network or execution error', 'error');
    } finally {
        if (runBtn) runBtn.classList.remove('hidden');
        if (stopBtn) stopBtn.classList.add('hidden');
        
        // Remove spinner from header
        const spinner = resultsContent.querySelector('.animate-spin');
        if(spinner) spinner.parentElement.remove();
        
        const copyBtn = document.getElementById('copy-output-btn');
        if(copyBtn) copyBtn.classList.remove('hidden');
    }
}

async function stopTask(taskName) {
    const taskId = taskName.replace(/\./g, '_');
    const stopBtn = document.getElementById(`btn-stop-${taskId}`);
    
    if(stopBtn) {
        stopBtn.disabled = true;
        stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Stopping...';
    }
    
    try {
        const response = await fetch(`/stop_task/${taskName}`, { method: 'POST' });
        const data = await response.json();
        showToast(data.message, data.status);
    } catch (error) {
        showToast('Error stopping task', 'error');
    } finally {
        if(stopBtn) {
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<i class="fas fa-stop mr-2"></i> Stop';
        }
    }
}

// ============== PLOT BROWSER LOGIC ==============
let currentPath = "";

async function loadLatestPlot() {
    try {
        const response = await fetch('/latest_plot');
        const data = await response.json();
        
        const container = document.getElementById('latest-container');
        if (data.url) {
            // Add a timestamp to bypass browser cache
            const timestamp = new Date().getTime();
            document.getElementById('latest-img').src = `${data.url}?t=${timestamp}`;
            document.getElementById('latest-name').textContent = data.filename + ' (' + new Date(data.timestamp * 1000).toLocaleString() + ')';
            container.classList.remove('hidden');
        } else {
            container.classList.add('hidden');
        }
    } catch (e) {
        console.error("Error loading latest plot", e);
    }
}

async function browsePath(path) {
    const contentDiv = document.getElementById('browser-content');
    // Don't show loading spinner if refreshing current path (smooth update)
    // contentDiv.innerHTML = '...'; 
    
    try {
        const encodedPath = path.split('/').map(c => encodeURIComponent(c)).join('/');
        const response = await fetch(`/browse_plots?path=${encodedPath}`);
        const data = await response.json();

        if (data.error) {
            contentDiv.innerHTML = `<div class="col-span-full text-center py-8 text-red-500">Error: ${data.error}</div>`;
            return;
        }

        currentPath = data.current_path;
        document.getElementById('current-path').textContent = currentPath || '/';
        document.getElementById('btn-up').disabled = !currentPath;
        if (!currentPath) document.getElementById('btn-up').classList.add('opacity-50', 'cursor-not-allowed');
        else document.getElementById('btn-up').classList.remove('opacity-50', 'cursor-not-allowed');

        contentDiv.innerHTML = '';

        if (data.dirs.length === 0 && data.files.length === 0) {
            contentDiv.innerHTML = '<div class="col-span-full text-center py-8 text-gray-400">Empty directory</div>';
            return;
        }

        // Render Directories
        data.dirs.forEach(dir => {
            const el = document.createElement('div');
            el.className = 'bg-blue-50 hover:bg-blue-100 text-blue-700 p-3 rounded-xl cursor-pointer flex flex-col items-center justify-center transition-all duration-200 border border-blue-100 hover:shadow-md h-24';
            el.onclick = () => browsePath(currentPath ? `${currentPath}/${dir}` : dir);
            el.innerHTML = `<div class="text-3xl mb-1">📁</div><div class="font-semibold text-xs text-center truncate w-full px-1">${dir}</div>`;
            contentDiv.appendChild(el);
        });

        // Render Files
        data.files.forEach(file => {
            const el = document.createElement('div');
            el.className = 'group relative bg-gray-100 rounded-xl overflow-hidden shadow-sm hover:shadow-lg transition-all duration-200 aspect-video cursor-zoom-in border border-gray-200';
            const safeDir = currentPath ? currentPath + '/' : '';
            const fullUrl = `/img/${safeDir}${file}`;
            const timestamp = new Date().getTime(); // Cache busting
            
            el.onclick = () => openModal(fullUrl);
            el.innerHTML = `
                <img src="${fullUrl}?t=${timestamp}" loading="lazy" alt="${file}" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110">
                <div class="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs p-2 truncate text-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">${file}</div>
            `;
            contentDiv.appendChild(el);
        });

    } catch (error) {
        contentDiv.innerHTML = `<div class="col-span-full text-center py-8 text-red-500">Network Error: ${error.message}</div>`;
    }
}

function navigateUp() {
    if (!currentPath) return;
    const parts = currentPath.split('/');
    parts.pop();
    browsePath(parts.join('/'));
}

async function refreshPlots() {
    await Promise.all([loadLatestPlot(), browsePath(currentPath)]);
}