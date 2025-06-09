# 🧠 AI 3D Generator with Intelligent Memory System

An advanced AI-powered application that transforms text prompts into 3D models through a complete text-to-image-to-3D pipeline. Features intelligent memory system with semantic search, natural language query capabilities, and dual GUI interfaces. Built for production deployment with Poetry package management and offline model support.

**Designed for Openfabric's AI Developer Challenge - demonstrating robust infrastructure, modular design, and creative AI pipeline integration.**

---

## 🚀 Features

- 🎨 **Complete AI Pipeline**: Text → Enhanced Prompt → Image → 3D Model generation
- 🧠 **Intelligent Memory System**: Stores all generations with semantic search using FAISS and sentence transformers
- 💬 **Natural Language Memory Queries**: "Show me my robot from yesterday"
- 🎛️ **Dual GUI Support**:
  - **Chainlit GUI** (conversational interface with memory integration)
  - **Gradio GUI** (visual interface with memory capabilities)
- 🌐 **Openfabric Integration**: Uses Text-to-Image and Image-to-3D apps via SDK
- 🤖 **Local LLM Integration**: DeepSeek for prompt enhancement and understanding
- 📦 **Poetry-based** dependency management with optimized package selection
- 🔍 **Semantic Search**: Vector-based similarity search with keyword fallback
- 📊 **Memory Analytics**: Track creation history, usage patterns, and quality metrics

---

## ⚙️ Prerequisites

- **Python 3.9–3.10** (required for Chainlit compatibility)
- [Poetry](https://python-poetry.org/) for dependency management
- **4GB+ RAM** (for sentence transformer models and FAISS indexing)

---

## 🧪 Setup and Installation

```bash
# Step 0: Navigate to the project root
cd ai-test

# Step 0.5: (Optional) Deactivate any global or existing virtual environment
deactivate # if already active

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

## 🌐 Access Your Application

Once running, access your AI 3D generator at:

- **Chainlit GUI**: `http://localhost:7860` (conversational interface)
- **Gradio GUI**: `http://localhost:7860` (visual interface)
- **API Server**: `http://localhost:8888`
- **Swagger Documentation**: `http://localhost:8888/swagger-ui/#/`

---

## 🧠 Memory System Functionality

### **How Memory Works**

The application implements a sophisticated memory system that stores and recalls all AI generations:

#### **Short-Term Memory (Session Context)**
- Maintains conversation context during active sessions
- Tracks user preferences and recent interactions
- Enables contextual prompt enhancement

#### **Long-Term Memory (Persistent Storage)**
- **SQLite Database**: Stores all generation metadata, prompts, and file paths
- **FAISS Vector Index**: Enables semantic similarity search across memories
- **Sentence Transformers**: Converts text to embeddings for intelligent retrieval

### **Memory Features**

#### **Automatic Storage**
Every generation is automatically stored with:
- Original user prompt
- Enhanced prompt (via DeepSeek LLM)
- Generated image path
- 3D model path
- Timestamp and quality metrics
- User session information

#### **Natural Language Queries**
Users can query their memory using natural language:

**Generation Commands:**
```
"A futuristic robot with glowing blue eyes"
"Medieval castle on a floating island"
"Steampunk airship with brass details"
```

**Memory Query Examples:**
```
"Show me my recent creations"
"Find my robot from yesterday"
"List my work from this week"
```

#### **Semantic Search**
- **Vector Similarity**: Finds similar creations based on meaning, not just keywords
- **Temporal Queries**: Search by time periods ("yesterday", "last week")
- **Entity Recognition**: Find by objects, colors, styles
- **Hybrid Search**: Combines semantic and keyword matching

### **Memory Architecture**

```

Memory System Components:
├── memory_manager.py          \# Core storage and retrieval operations
├── conversation_tracker.py    \# NLP for parsing memory queries
├── semantic_search.py         \# FAISS-powered similarity search
└── ai_memory.db              \# SQLite database with FAISS indexing

```

---

## 🔄 AI Pipeline Workflow

### **Complete Generation Process**

1. **User Input**: Text prompt via GUI
2. **Memory Context**: System checks for relevant past creations
3. **LLM Enhancement**: DeepSeek expands prompt with creative details
4. **Image Generation**: Openfabric Text-to-Image app creates visual
5. **3D Conversion**: Openfabric Image-to-3D app generates 3D model
6. **Memory Storage**: All results stored with semantic indexing
7. **User Delivery**: Files saved and displayed in GUI

### **Openfabric Integration**

The application uses two Openfabric apps in sequence:
- **Text-to-Image**: `c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network`
- **Image-to-3D**: `5891a64fe34041d98b0262bb1175ff07.node3.openfabric.network`

---

## 🧹 Troubleshooting and Debugging

### **Common Issues**

#### **Port Conflicts**
```bash
# Check what's using ports
sudo lsof -i:7860
sudo lsof -i:8888

# Free up ports if needed
sudo fuser -k 8888/tcp
sudo fuser -k 7860/tcp
```

#### **Memory System Issues**
```bash
# Check memory database
ls -la app/memory/
sqlite3 app/memory/ai_memory.db ".tables"

# Verify FAISS installation
python3 -c "import faiss; print('FAISS working')"

# Test sentence transformers
python3 -c "from sentence_transformers import SentenceTransformer; print('Models working')"
```

#### **LLM Issues**
```bash
# Check if DeepSeek/Ollama is running
curl -s http://localhost:11434/api/version

# List available models
ollama list

# Re-download model if needed
ollama pull deepseek-r1:1.5b
```

### **Debug Logs Location**
- **Application logs**: `./app/logs/`
- **Memory database**: `./app/memory/`
- **Generated outputs**: `./app/outputs/`

---

## 🏆 Technical Highlights

- **Advanced Memory Architecture**: Combines SQLite persistence with FAISS vector indexing
- **Semantic Understanding**: Natural language processing for intuitive memory queries
- **Openfabric SDK Integration**: Proper use of `Stub`, `Remote`, schema, and manifest
- **Local LLM Processing**: DeepSeek integration for creative prompt enhancement
- **Modular Design**: Clean separation of concerns with extensible architecture
- **User Experience**: Dual GUI approach for different interaction preferences
- **Performance Optimized**: Pre-downloaded models and efficient vector search

---

## 📋 Example Usage

### **Basic Generation**
```
User: "Create a glowing dragon standing on a cliff at sunset"
→ LLM enhances to detailed visual description
→ Text-to-Image generates stunning artwork
→ Image-to-3D creates interactive 3D model
→ System remembers for future reference
```

### **Memory-Driven Creation**
```
User: "Make a robot like the one from last week, but with wings"
→ System searches memory for previous robots
→ Uses similar creation as context for enhancement
→ Generates new robot with requested modifications
→ Stores relationship to original creation
```

Built with ❤️ for the Openfabric AI Developer Challenge