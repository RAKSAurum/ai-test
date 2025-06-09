# 🚀 AI 3D Generator with Intelligent Memory System

An advanced AI-powered application that transforms text descriptions into 3D models using an integrated text-to-image-to-3D pipeline. Features a sophisticated memory system with semantic search capabilities, natural language queries, and dual GUI interfaces (Chainlit and Gradio).

## ✨ Key Features

- 🎨 **Complete Generation Pipeline**
  - Text prompt enhancement via DeepSeek LLM
  - Image generation through Openfabric
  - Automatic 3D model conversion (GLB) through Openfabric
  - Downloadable assets with proper organization

- 🧠 **Intelligent Memory System**
  - Semantic search using FAISS vectors
  - Natural language memory queries
  - Automatic context enhancement
  - Deduplication of similar content

- 🎛️ **Dual Interface Support**
  - **Chainlit GUI**: Conversational interface with rich markdown
  - **Gradio GUI**: Visual interface with live preview
  - Both with full memory integration

## 🛠️ Prerequisites

- Python 3.9-3.10 (required for Chainlit)
- Poetry for dependency management
- 4GB+ RAM recommended

## 📦 Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
cd app
pip install poetry
poetry install
```

## 🚀 Running the Application

```bash
# Launch Chainlit GUI (recommended)
bash start.sh 
1

# OR launch Gradio GUI
bash start.sh
2
```

Access the interfaces at:
- Chainlit: http://localhost:7860
- Gradio: http://localhost:7860 
- API: http://localhost:8888

## 💡 Usage Examples

### Generate a 3D Model
```
"Create a futuristic robot with glowing blue eyes and metallic armor"
"A medieval castle on a floating island with waterfalls"
"Steampunk mechanical dragon with brass gears"
```

### Query Your Memory
```
"Show me my recent creations"
"Find my robot from yesterday"
"Create something like my last dragon"
```

## 🧠 Memory System Architecture

```
memory/
├── __init__.py              # System initialization
├── memory_manager.py        # Core storage/retrieval
├── conversation_tracker.py  # Query understanding
├── semantic_search.py      # FAISS search
└── ai_memory.db           # SQLite database
```

### Memory Features
- Automatic storage of all generations
- Semantic similarity search
- Natural language queries
- Deduplication of similar content
- Context-aware generation

## 🔧 Troubleshooting

### Port Conflicts
```bash
# Check ports
sudo lsof -i:7860
sudo lsof -i:8888

# Free if needed
sudo fuser -k 7860/tcp
sudo fuser -k 8888/tcp
```

### File Permissions
```bash
# Fix permissions
chmod 755 downloads outputs logs memory
chmod 644 downloads/* outputs/* logs/*
```

### Memory System
```bash
# Verify database
sqlite3 memory/ai_memory.db ".tables"

# Check FAISS
python3 -c "import faiss; print('FAISS OK')"
```

## 🏷️ Version
Current Version: 1.0.0