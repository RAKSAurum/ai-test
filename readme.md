# 🧠 AI Developer Challenge: Intelligent Assistant

An AI-powered, multimodal assistant that integrates advanced natural language understanding, knowledge retrieval, and dynamic user interfaces (Gradio & Chainlit) to deliver seamless user experiences. Designed for Openfabric's AI Developer Challenge, this project demonstrates robust infrastructure, modular design, and scalable deployment.

---

## 🚀 Features

- 🔎 **Semantic Intelligence**: Natural language understanding and contextual Q&A
- 🎛️ **Dual GUI Support**: Choose between:
  - **Gradio GUI** (quick prototyping and visual interface)
  - **Chainlit GUI** (conversational UI with memory)
- 📦 **Poetry**-based Python dependency management
- 🧠 **Modular Code**: Easy to extend and adapt

---

## ⚙️ Prerequisites

- Python 3.9–3.10
- [Poetry](https://python-poetry.org/)

---

## 🧪 Local Setup

```bash

# Step 0: Navigate to the project root

cd ai-test

# Step 0.5: (Optional) Deactivate any global or existing virtual environment

deactivate  \# if already active

# Step 0.6: Create and activate a local virtual environment in this folder

python3 -m venv .venv
source .venv/bin/activate

# Step 1: Navigate to the app directory

cd app

# Step 2: Install Poetry and dependencies

pip install poetry
poetry lock
poetry install

# Step 3: Launch core engine

poetry run python3 ignite.py -w --port 8888

# Step 4a: Launch Chainlit GUI (on port 7860)

poetry run chainlit run chainlit_gui.py -w --port 7860

# OR

# Step 4b: Launch Gradio GUI

poetry run python3 gradio_gui.py

```

> ⚠️ **Important:** All setup and execution should happen inside a local `.venv` created within `ai-test/`. Do **not** use global Python or Poetry installations.

---

## 🧹 Troubleshooting

- Make sure ports `8888` and `7860` are free:
```

sudo fuser -k 8888/tcp
sudo fuser -k 7860/tcp

```
    
- Logs and memory data are stored in:
```

./app/logs/
./app/memory/
./app/outputs/

```