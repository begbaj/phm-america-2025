# 📖 Guida alla Modifica dei Parametri della Web GUI

Questa guida spiega come gestire i parametri visualizzati nella Web GUI per i vari task. La GUI è dinamica e si basa su file di configurazione JSON.

## 📂 Posizione dei File
Ogni script contenuto nella cartella `tasks/` (es. `mio_task.py`) può avere un file di configurazione in:
`web_gui/task_configs/mio_task.json`

## ➕ Aggiungere un Parametro
Per aggiungere un parametro, apri il file `.json` corrispondente e inserisci un nuovo oggetto nell'array `params` all'interno di uno dei `groups`.

### Tipi di Campi Supportati

#### 🔢 Numero (`number`)
```json
{
    "id": "soglia_valore",
    "label": "Soglia",
    "type": "number",
    "default": 0.5,
    "step": 0.1
}
```

#### 📝 Testo (`text`)
```json
{
    "id": "nome_esperimento",
    "label": "Nome Esperimento",
    "type": "text",
    "default": "test_01"
}
```

#### 🔽 Selezione Singola (`select`)
```json
{
    "id": "metodo",
    "label": "Metodo",
    "type": "select",
    "options": ["opzione_a", "opzione_b", "opzione_c"],
    "default": "opzione_a"
}
```

#### 🔘 Checkbox (`checkbox`)
*Viene inviato allo script solo se selezionato (True).*
```json
{
    "id": "abilita_filtro",
    "label": "Abilita Filtro",
    "type": "checkbox",
    "default": false
}
```

#### 📑 Selezione Multipla (`multiselect`)
*Invia una stringa separata da virgole (es. `--passaggi step1,step2`).*
```json
{
    "id": "passaggi",
    "label": "Passaggi da Eseguire",
    "type": "multiselect",
    "options": ["carica", "elabora", "salva"],
    "default": ["carica", "elabora"]
}
```

## ➖ Rimuovere un Parametro
Per rimuovere un campo, è sufficiente eliminare il relativo blocco JSON dall'array `params`.

## 🛠️ Integrazione con lo Script Python
La GUI invia i parametri come argomenti CLI trasformando i nomi con il prefisso `--` e sostituendo gli underscore `_` con trattini `-`.

**Esempio:**
Se nel JSON hai `"id": "outlier_threshold"`, la GUI eseguirà:
`python tasks/script.py --outlier-threshold 0.08`

Assicurati che il tuo script Python gestisca gli argomenti usando `argparse`:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outlier-threshold", type=float, default=0.08)
args = parser.parse_args()
```

## 🎨 Organizzazione in Gruppi
Puoi organizzare i parametri in sezioni logiche modificando l'array `groups`. Ogni gruppo ha un `title` e una lista di `params`.
