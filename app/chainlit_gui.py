import chainlit as cl
import os
import time
import shutil
import requests
import json

# Add path for memory imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import memory system components
from memory import MemoryManager, ConversationTracker, SemanticSearch

# Configuration
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Initialize memory system (completely local)
memory_manager = MemoryManager()
conversation_tracker = ConversationTracker()
semantic_search = SemanticSearch()

# Global session tracking
user_sessions = {}

def call_api_sync(prompt):
    """Synchronous API call function to be wrapped with cl.make_async"""
    api_payload = {"prompt": prompt}
    
    response = requests.post(
        "http://localhost:8888/execution",
        json=api_payload,
        timeout=300  # 5 minutes timeout
    )
    
    return response

@cl.on_chat_start
async def start():
    """Initialize chat with welcome message, memory system, and quick action examples."""
    # Initialize user session with memory
    user_id = cl.user_session.get("id", "default")
    session_id = memory_manager.create_session(user_id)
    
    # Store session info
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", user_id)
    user_sessions[user_id] = session_id
    
    # Get recent memories for context
    recent_memories = memory_manager.get_recent_memories(user_id, limit=3)
    
    # Build welcome message with memory context
    welcome_content = """# 🚀 AI 3D Generator with Memory
**Transform text into 3D models with intelligent memory recall**

**How it works:**
1. 🧠 Enhanced prompts using DeepSeek LLM
2. 🎨 Queue-based image generation via Openfabric
3. 🗿 Direct 3D model conversion with GLB output
4. 📥 Automatic file downloads
5. 💾 **Memory system remembers everything!**

**Memory Commands:**
- "Show me what I created yesterday"
- "Find my robot from last week"
- "Create something like my last dragon"
- "List my recent creations"

Just describe what you want to create!"""
    
    if recent_memories:
        welcome_content += f"\n\n**Your Recent Creations:**"
        for i, memory in enumerate(recent_memories, 1):
            timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(memory.timestamp))
            welcome_content += f"\n{i}. {memory.original_prompt} ({timestamp})"
    
    await cl.Message(content=welcome_content).send()
    
    # Add quick action buttons
    actions = [
        cl.Action(name="robot", label="🤖 Robot Example", payload={"prompt": "A futuristic robot with glowing blue eyes and metallic armor"}),
        cl.Action(name="castle", label="🏰 Castle Example", payload={"prompt": "A medieval castle on a floating island with waterfalls"}),
        cl.Action(name="dragon", label="🐉 Dragon Example", payload={"prompt": "A steampunk mechanical dragon with brass gears and copper wings"}),
        cl.Action(name="memory_search", label="🧠 Search Memory", payload={"prompt": "show me my recent creations"}),
        cl.Action(name="similar_creation", label="🔄 Create Similar", payload={"prompt": "create something like my last creation but with wings"})
    ]
    
    await cl.Message(content="**Quick Actions:**", actions=actions).send()

@cl.action_callback("robot")
async def on_action_robot(action):
    await handle_generation(action.payload["prompt"])

@cl.action_callback("castle")
async def on_action_castle(action):
    await handle_generation(action.payload["prompt"])

@cl.action_callback("dragon")
async def on_action_dragon(action):
    await handle_generation(action.payload["prompt"])

@cl.action_callback("memory_search")
async def on_action_memory_search(action):
    await handle_memory_query(action.payload["prompt"])

@cl.action_callback("similar_creation")
async def on_action_similar_creation(action):
    await handle_generation(action.payload["prompt"])

@cl.on_message
async def main(message: cl.Message):
    """Main message handler for user input with memory integration."""
    user_input = message.content
    
    # Parse input to determine if it's a memory query
    parsed_query = conversation_tracker.parse_memory_query(user_input)
    
    if parsed_query['is_memory_query'] and parsed_query['intent'] in ['recall', 'list']:
        await handle_memory_query(user_input)
    else:
        await handle_generation(user_input)

async def handle_memory_query(query: str):
    """Handle memory-related queries with natural language processing."""
    user_id = cl.user_session.get("user_id", "default")
    
    # Show processing message
    processing_msg = cl.Message(content="🧠 Searching through your memories...")
    await processing_msg.send()
    
    try:
        # Parse the memory query
        parsed_query = conversation_tracker.parse_memory_query(query)
        search_query = conversation_tracker.build_search_query(parsed_query)
        
        # Perform search based on query type
        if search_query['time_range']:
            search_results = semantic_search.temporal_search(user_id, search_query['time_range'], limit=10)
        elif search_query['entity_filters'] or search_query['text_search']:
            search_results = semantic_search.hybrid_search(
                search_query['text_search'], user_id, search_query.get('time_range'), 
                search_query.get('entity_filters'), limit=10
            )
        else:
            recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
            search_results = [
                semantic_search.SearchResult(memory_id=mem.id, relevance_score=1.0, match_type='recent', matched_content=mem.original_prompt) 
                for mem in recent_memories
            ]
        
        # Get full memory details and create response
        if search_results:
            memories = []
            elements = []
            
            for result in search_results[:5]:
                with memory_manager._get_connection() as conn:
                    cursor = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (result.memory_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        memory = memory_manager._row_to_memory_entry(row)
                        memories.append(memory)
                        
                        if memory.image_path and os.path.exists(memory.image_path):
                            elements.append(cl.Image(name=f"Image: {memory.original_prompt[:30]}...", path=memory.image_path, display="inline"))
                        
                        if memory.model_path and os.path.exists(memory.model_path):
                            elements.append(cl.File(name=f"3D Model: {memory.original_prompt[:30]}...", path=memory.model_path, display="inline"))
            
            # Format response
            response_text = conversation_tracker.format_memory_response(memories, parsed_query.get('intent', 'recall'))
            
            processing_msg.content = response_text
            processing_msg.elements = elements
            await processing_msg.update()
            
            if memories:
                actions = [cl.Action(name="create_similar_to_memory", label="🎨 Create Similar", payload={"memory_id": memories[0].id, "prompt": memories[0].original_prompt})]
                await cl.Message(content="**Actions:**", actions=actions).send()
        else:
            processing_msg.content = "🤔 I couldn't find any memories matching your request. Try describing what you're looking for differently, or create something new!"
            await processing_msg.update()
    
    except Exception as e:
        processing_msg.content = f"❌ Memory search failed: {str(e)}"
        await processing_msg.update()

@cl.action_callback("create_similar_to_memory")
async def on_create_similar_to_memory(action):
    memory_id = action.payload["memory_id"]
    original_prompt = action.payload["prompt"]
    variation_prompt = f"Create a variation of: {original_prompt}"
    await handle_generation(variation_prompt, reference_memory_id=memory_id)

def copy_file_safely(src, dest):
    """Safely copy file with proper binary handling for GLB files."""
    try:
        if src and os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(src, 'rb') as src_file:
                with open(dest, 'wb') as dest_file:
                    shutil.copyfileobj(src_file, dest_file)
            return dest
    except Exception as e:
        print(f"Error copying file {src} to {dest}: {e}")
    return None

def get_proper_extension(file_path):
    """Determine the proper file extension based on file content."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
            if header[:4] == b'glTF':
                return '.glb'
            else:
                f.seek(0)
                try:
                    content = f.read(500).decode('utf-8', errors='ignore')
                    if '"asset"' in content and '"version"' in content and '"generator"' in content:
                        return '.gltf'
                except:
                    pass
                return '.glb'
    except Exception:
        return '.gltf'

async def handle_generation(prompt: str, reference_memory_id: str = None):
    """Core generation workflow calling your API directly with memory integration."""
    start_time = time.time()
    user_id = cl.user_session.get("user_id", "default")
    session_id = cl.user_session.get("session_id")
    
    # Initialize processing message
    processing_msg = cl.Message(content="")
    await processing_msg.send()
    
    try:
        # Memory Context Retrieval
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 1/3:** Analyzing memory context...
**Status:** Searching for relevant past creations"""
        await processing_msg.update()
        
        # Get memory context for prompt enhancement
        memory_context = []
        enhanced_prompt = prompt
        
        if reference_memory_id:
            with memory_manager._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (reference_memory_id,))
                row = cursor.fetchone()
                if row:
                    memory_context.append(memory_manager._row_to_memory_entry(row))
        else:
            search_results = semantic_search.semantic_search(prompt, user_id, limit=3)
            for result in search_results:
                memories = memory_manager.search_memories(result.matched_content, user_id, limit=1)
                memory_context.extend(memories)
        
        # Enhance prompt locally using memory context
        if memory_context:
            context_prompts = [mem.original_prompt for mem in memory_context[:2]]
            enhanced_prompt = f"{prompt} (inspired by: {', '.join(context_prompts)})"
        
        # API Call with cl.make_async to prevent timeout
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 2/3:** Calling API for generation...
**Status:** This may take 1-2 minutes for high-quality results
**Memory Context:** {len(memory_context)} relevant memories found"""
        await processing_msg.update()
        
        # FIXED: Use cl.make_async to prevent 30-second timeout
        response = await cl.make_async(call_api_sync)(enhanced_prompt)
        
        # File Management and Memory Storage
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 3/3:** Processing results and storing in memory...
**Status:** Organizing generated files"""
        await processing_msg.update()
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "Generation completed")
            
            # Find generated files
            image_files = []
            model_files = []
            
            if os.path.exists("outputs"):
                for file in os.listdir("outputs"):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(f"outputs/{file}")
                    elif file.endswith(('.glb', '.obj', '.ply', '.stl')):
                        model_files.append(f"outputs/{file}")
            
            # Get latest files
            latest_image = max(image_files, key=os.path.getctime) if image_files else None
            latest_model = max(model_files, key=os.path.getctime) if model_files else None
            
            # Prepare files for display
            elements = []
            download_info = []
            timestamp = int(time.time())
            
            # Handle image
            image_path = None
            if latest_image and os.path.exists(latest_image):
                ext = os.path.splitext(latest_image)[1] or '.png'
                image_filename = f"image_{timestamp}{ext}"
                image_path = os.path.join(DOWNLOADS_DIR, image_filename)
                
                if copy_file_safely(latest_image, image_path):
                    elements.append(cl.Image(name=image_filename, path=image_path, display="inline", size="large"))
                    download_info.append(f"🖼️ **Image:** `{image_filename}`")
            
            # Handle 3D model
            model_path = None
            if latest_model and os.path.exists(latest_model):
                proper_ext = get_proper_extension(latest_model)
                model_filename = f"model_{timestamp}{proper_ext}"
                model_path = os.path.join(DOWNLOADS_DIR, model_filename)
                
                if copy_file_safely(latest_model, model_path):
                    mime_type = "model/gltf-binary" if proper_ext == '.glb' else "model/gltf+json"
                    elements.append(cl.File(name=model_filename, path=model_path, display="inline", mime=mime_type))
                    download_info.append(f"🗿 **3D Model:** `{model_filename}`")
                    
                    file_size = os.path.getsize(model_path)
                    download_info.append(f"   └── Size: {file_size:,} bytes")
                    download_info.append(f"   └── Format: {proper_ext.upper()}")
            
            # Store in memory
            try:
                memory_metadata = {
                    'generation_time': time.time(),
                    'processing_time': time.time() - start_time,
                    'reference_memory_id': reference_memory_id,
                    'memory_context_used': len(memory_context) > 0,
                    'api_response': message
                }
                
                memory_id = memory_manager.store_generation(
                    session_id=session_id, user_id=user_id, original_prompt=prompt,
                    enhanced_prompt=enhanced_prompt, image_path=image_path, 
                    model_path=model_path, metadata=memory_metadata
                )
                
                semantic_search.add_memory_embedding(memory_id, f"{prompt} {enhanced_prompt}")
            except Exception as e:
                print(f"Failed to store in memory: {e}")
            
            # Success message
            total_time = int(time.time() - start_time)
            success_content = f"""## ✅ Generation Complete! ({total_time}s)

**Prompt:** "{prompt}"
**Memory Context:** {len(memory_context)} relevant memories used

**Generated Files:**
{chr(10).join(download_info)}

**Downloads Location:** `{os.path.abspath(DOWNLOADS_DIR)}`"""
            
            await cl.Message(content=success_content, elements=elements).send()
            
            # Usage tips
            await cl.Message(content="""**💾 Memory:** This creation has been stored and can be recalled using natural language queries!

**3D File Usage:**
- 🔧 **Blender:** File → Import → glTF 2.0 (.glb/.gltf)
- 🎮 **Unity:** Drag file into Assets folder
- 🌐 **Online:** gltf-viewer.donmccurdy.com""").send()
            
            # Next actions
            actions = [
                cl.Action(name="generate_another", label="🎨 Generate Another", payload={"action": "new"}),
                cl.Action(name="create_variation", label="🔄 Create Variation", payload={"prompt": f"create a variation of: {prompt}"}),
                cl.Action(name="view_memories", label="🧠 View My Memories", payload={"prompt": "show me my recent creations"})
            ]
            
            await cl.Message(content="**What's next?**", actions=actions).send()
        
        else:
            processing_msg.content = f"""## ❌ Generation Failed

**Error:** HTTP {response.status_code}
**Response:** {response.text[:200]}

**Your API is running but returned an error. Check the server logs.**"""
            await processing_msg.update()
        
    except requests.exceptions.ConnectionError:
        processing_msg.content = f"""## ❌ Connection Failed

**Error:** Cannot connect to API server at http://localhost:8888

**Please check:**
1. Is your API server running?
2. Try: `python main.py` to start the server"""
        await processing_msg.update()
        
    except Exception as e:
        processing_msg.content = f"""## ❌ Generation Failed

**Error:** {str(e)}

**Memory Commands still work:**
- "Show me my recent creations"
- "Find my robot from yesterday\""""
        await processing_msg.update()

@cl.action_callback("generate_another")
async def generate_another(action):
    await cl.Message(content="Great! Just type your new description and I'll generate another 3D model for you. 🎨").send()

@cl.action_callback("create_variation")
async def create_variation(action):
    await handle_generation(action.payload["prompt"])

@cl.action_callback("view_memories")
async def view_memories(action):
    await handle_memory_query(action.payload["prompt"])

if __name__ == "__main__":
    cl.run()