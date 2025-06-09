"""
AI 3D Generator with Memory Integration

A Chainlit-based application that provides AI-powered 3D model generation
with intelligent memory management and natural language query capabilities.
"""

import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Optional, Set, Dict, List, TypeVar, Generic
from datetime import datetime

import chainlit as cl
import requests

# Add path for memory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from memory import MemoryManager, ConversationTracker, SemanticSearch
from memory.semantic_search import SearchResult
from memory import MemoryEntry

# Configure logging with safe directory creation
import logging

# Safely create logging handlers
log_handlers = [logging.StreamHandler()]

try:
    os.makedirs("logs", exist_ok=True)
    log_handlers.append(logging.FileHandler('logs/chainlit.log'))
except Exception as e:
    print(f"Warning: Could not create log file: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class DedupManager(Generic[T]):
    """Generic deduplication manager for memory results."""
    
    def __init__(self, name: str):
        self.name = name
        self.seen_items: Set[str] = set()
        self.dedup_stats: Dict[str, int] = {
            'total_processed': 0,
            'duplicates_filtered': 0,
            'unique_passed': 0
        }
        try:
            os.makedirs("logs", exist_ok=True)
            self._setup_logging()
        except Exception as e:
            print(f"Warning: Failed to setup logging: {e}")
            self.logger = logging.getLogger(f"dedup_{self.name}")

    def _setup_logging(self) -> None:
        self.logger = logging.getLogger(f"dedup_{self.name}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"logs/dedup_{self.name}_{datetime.now().strftime('%Y%m%d')}.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def deduplicate(self, items: List[T], key_func) -> List[T]:
        """Deduplicate items based on a key function."""
        self.dedup_stats['total_processed'] += len(items)
        
        seen = set()
        deduped = []
        
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
                self.dedup_stats['unique_passed'] += 1
            else:
                self.dedup_stats['duplicates_filtered'] += 1
                self.logger.info(f"Filtered duplicate: {key}")
        
        return deduped


# Configuration
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Initialize memory system (completely local)
memory_manager = MemoryManager()
conversation_tracker = ConversationTracker()
semantic_search = SemanticSearch()

# Global session tracking
user_sessions = {}

# Initialize dedup manager globally
memory_dedup = DedupManager[MemoryEntry]("chainlit")


def call_api_sync(prompt: str) -> requests.Response:
    """
    Synchronous API call function for 3D model generation.
    
    Makes a POST request to the local API server for 3D model generation
    with the provided prompt text.
    
    Args:
        prompt (str): Text prompt for 3D model generation.
        
    Returns:
        requests.Response: HTTP response from the generation API.
        
    Raises:
        requests.exceptions.RequestException: If the API call fails.
    """
    api_payload = {"prompt": prompt}
    
    response = requests.post(
        "http://localhost:8888/execution",
        json=api_payload,
        timeout=300  # 5 minutes timeout
    )
    
    return response


async def debug_memory() -> list:
    """
    Debug function to test memory system functionality directly.
    
    Performs comprehensive testing of the memory system including
    database connections, recent memory retrieval, and conversation tracking.
    
    Returns:
        list: List of recent memory entries for debugging purposes.
    """
    user_id = cl.user_session.get("user_id", "default")
    
    try:
        print(f"DEBUG: Current user_id: {user_id}")
        print(f"DEBUG: Session user_id: {cl.user_session.get('id', 'none')}")
        
        # Test 1: Check recent memories directly
        recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
        print(f"DEBUG: Found {len(recent_memories)} recent memories")
        
        for i, memory in enumerate(recent_memories):
            print(f"DEBUG Memory {i}: {memory.original_prompt} (ID: {memory.id})")
        
        # Test 2: Check database connection
        with sqlite3.connect(memory_manager.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_entries WHERE user_id = ?", (user_id,))
            count = cursor.fetchone()[0]
            print(f"DEBUG: Database has {count} entries for user {user_id}")
            
            # Check all users in database
            cursor = conn.execute("SELECT DISTINCT user_id, COUNT(*) FROM memory_entries GROUP BY user_id")
            all_users = cursor.fetchall()
            print(f"DEBUG: All users in database: {all_users}")
        
        # Test 3: Check conversation tracker
        parsed_query = conversation_tracker.parse_memory_query("show me my recent creations")
        print(f"DEBUG: Parsed query: {parsed_query}")
        
        return recent_memories
        
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


async def debug_file_system() -> None:
    """Debug function to check file system state."""
    print("=== FILE SYSTEM DEBUG ===")
    
    # Check directories
    for directory in ["outputs", "downloads"]:
        if os.path.exists(directory):
            files = os.listdir(directory)
            print(f"{directory}/: {len(files)} files")
            for file in files[:5]:  # Show first 5 files
                filepath = os.path.join(directory, file)
                size = os.path.getsize(filepath)
                accessible = verify_file_permissions(filepath)
                print(f"  - {file}: {size} bytes, accessible: {accessible}")
        else:
            print(f"{directory}/: Does not exist")
    
    # Check recent memory entries
    user_id = cl.user_session.get("user_id", "default")
    recent_memories = memory_manager.get_recent_memories(user_id, limit=3)
    print(f"\nRECENT MEMORIES: {len(recent_memories)}")
    
    for i, memory in enumerate(recent_memories):
        print(f"Memory {i+1}:")
        print(f"  - Prompt: {memory.original_prompt}")
        print(f"  - Image: {memory.image_path} (exists: {os.path.exists(memory.image_path) if memory.image_path else False})")
        print(f"  - Model: {memory.model_path} (exists: {os.path.exists(memory.model_path) if memory.model_path else False})")


@cl.on_chat_start
async def start() -> None:
    """
    Initialize chat session with welcome message and memory system setup.
    
    Creates a new user session, initializes the memory system, and displays
    a welcome message with recent creations and quick action buttons.
    """
    # Use consistent user ID instead of random session ID
    user_id = "default"
    session_id = memory_manager.create_session(user_id)
    
    print(f"DEBUG: Starting session for user_id: {user_id}, session_id: {session_id}")
    
    # Store session info
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", user_id)
    user_sessions[user_id] = session_id
    
    # Get recent memories for context
    recent_memories = memory_manager.get_recent_memories(user_id, limit=3)
    print(f"DEBUG: Found {len(recent_memories)} recent memories for welcome message")
    
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
    
    # Add quick action buttons with memory-aware examples
    actions = [
        cl.Action(
            name="robot", 
            label="🤖 Robot Example", 
            payload={"prompt": "A futuristic robot with glowing blue eyes and metallic armor"}
        ),
        cl.Action(
            name="castle", 
            label="🏰 Castle Example", 
            payload={"prompt": "A medieval castle on a floating island with waterfalls"}
        ),
        cl.Action(
            name="dragon", 
            label="🐉 Dragon Example", 
            payload={"prompt": "A steampunk mechanical dragon with brass gears and copper wings"}
        ),
        cl.Action(
            name="memory_search",
            label="🧠 Search Memory",
            payload={"prompt": "show me my recent creations"}
        ),
        cl.Action(
            name="similar_creation",
            label="🔄 Create Similar",
            payload={"prompt": "create something like my last creation but with wings"}
        ),
        cl.Action(
            name="debug_memory_direct",
            label="🔍 Debug Memory",
            payload={}
        ),
        cl.Action(
            name="debug_files",
            label="📁 Debug Files",
            payload={}
        )
    ]
    
    await cl.Message(
        content="**Quick Actions:**",
        actions=actions
    ).send()


@cl.action_callback("robot")
async def on_action_robot(action) -> None:
    """Handle robot example action callback."""
    await handle_generation(action.payload["prompt"])


@cl.action_callback("castle")
async def on_action_castle(action) -> None:
    """Handle castle example action callback."""
    await handle_generation(action.payload["prompt"])


@cl.action_callback("dragon")
async def on_action_dragon(action) -> None:
    """Handle dragon example action callback."""
    await handle_generation(action.payload["prompt"])


@cl.action_callback("memory_search")
async def on_action_memory_search(action) -> None:
    """Handle memory search action callback."""
    await handle_memory_query(action.payload["prompt"])


@cl.action_callback("similar_creation")
async def on_action_similar_creation(action) -> None:
    """Handle similar creation action callback."""
    await handle_generation(action.payload["prompt"])


@cl.action_callback("debug_memory_direct")
async def debug_memory_direct(action) -> None:
    """
    Direct memory test bypassing conversation tracker for debugging.
    
    Performs direct memory system testing and displays results with
    associated images and 3D model files if available.
    """
    user_id = cl.user_session.get("user_id", "default")
    
    try:
        print(f"DEBUG: Direct memory test for user: {user_id}")
        recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
        
        if recent_memories:
            response_text = f"Found {len(recent_memories)} memories:\n\n"
            elements = []
            
            for i, memory in enumerate(recent_memories, 1):
                timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(memory.timestamp))
                response_text += f"{i}. {memory.original_prompt} ({timestamp})\n"
                
                if memory.image_path and os.path.exists(memory.image_path):
                    elements.append(cl.Image(name=f"Image_{i}", path=memory.image_path, display="inline"))
                
                if memory.model_path and os.path.exists(memory.model_path):
                    elements.append(cl.File(name=f"Model_{i}", path=memory.model_path, display="inline"))
            
            await cl.Message(content=response_text, elements=elements).send()
        else:
            await cl.Message(content=f"No memories found for user: {user_id}").send()
            
    except Exception as e:
        await cl.Message(content=f"Direct memory test failed: {str(e)}").send()
        print(f"DEBUG: Direct memory test error: {e}")
        import traceback
        traceback.print_exc()


@cl.action_callback("debug_files")
async def debug_files_action(action) -> None:
    """Debug file system action callback."""
    await debug_file_system()
    await cl.Message(content="Check console for file system debug info").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    """
    Main message handler for user input with memory integration.
    
    Analyzes user input to determine if it's a memory query or generation request
    and routes to the appropriate handler function.
    
    Args:
        message (cl.Message): Incoming user message to process.
    """
    user_input = message.content
    
    # Parse input to determine if it's a memory query
    parsed_query = conversation_tracker.parse_memory_query(user_input)
    print(f"DEBUG: Input '{user_input}' parsed as: {parsed_query}")
    
    if parsed_query['is_memory_query'] and parsed_query['intent'] in ['recall', 'list']:
        await handle_memory_query(user_input)
    else:
        await handle_generation(user_input)


async def handle_memory_query(query: str) -> None:
    """Handle memory-related queries with comprehensive search and display."""
    user_id = cl.user_session.get("user_id", "default")
    print(f"DEBUG: Memory query from user {user_id}: '{query}'")
    
    # Show processing message
    processing_msg = cl.Message(content="🧠 Searching through your memories...")
    await processing_msg.send()
    
    try:
        # Parse the memory query
        parsed_query = conversation_tracker.parse_memory_query(query)
        print(f"DEBUG: Parsed query result: {parsed_query}")
        
        if not parsed_query['is_memory_query']:
            processing_msg.content = f"""Not recognized as memory query.
            
DEBUG INFO:
- Query: '{query}'
- Parsed: {parsed_query}
- Try: 'show me my recent creations' or 'list my recent work'"""
            await processing_msg.update()
            return
        
        search_query = conversation_tracker.build_search_query(parsed_query)
        print(f"DEBUG: Search query: {search_query}")
        
        # Determine search strategy based on intent
        if parsed_query['intent'] in ['recall', 'list']:
            print("DEBUG: Using recent memories for recall/list intent")
            recent_memories = memory_manager.get_recent_memories(user_id, limit=10)
            search_results = [
                SearchResult(
                    memory_id=mem.id,
                    relevance_score=1.0,
                    match_type='recent',
                    matched_content=mem.original_prompt
                ) for mem in recent_memories
            ]
        elif search_query['time_range']:
            print("DEBUG: Using temporal search")
            search_results = semantic_search.temporal_search(user_id, search_query['time_range'], limit=10)
        else:
            print("DEBUG: Using hybrid search")
            search_results = semantic_search.hybrid_search(
                search_query['text_search'], user_id, search_query.get('time_range'), 
                search_query.get('entity_filters'), limit=10
            )
            
            # If hybrid search returns nothing, fall back to recent memories
            if not search_results:
                print("DEBUG: Hybrid search returned 0 results, falling back to recent memories")
                recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
                search_results = [
                    SearchResult(
                        memory_id=mem.id,
                        relevance_score=1.0,
                        match_type='recent_fallback',
                        matched_content=mem.original_prompt
                    ) for mem in recent_memories
                ]
        
        logger.info(f"Found {len(search_results)} initial results")
        
        # Get full memory details and create response
        if search_results:
            memories = []
            elements = []
            seen_prompts = set()  # Track unique prompts
            
            for result in search_results:
                with sqlite3.connect(memory_manager.db_path) as conn:
                    cursor = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (result.memory_id,))
                    row = cursor.fetchone()
                    if row:
                        memory = memory_manager._row_to_memory_entry(row)
                        # Only add if we haven't seen this prompt before
                        if memory.original_prompt.lower().strip() not in seen_prompts:
                            memories.append(memory)
                            seen_prompts.add(memory.original_prompt.lower().strip())
                            
                            # Add visual elements for this memory
                            if memory.image_path and os.path.exists(memory.image_path):
                                elements.append(cl.Image(
                                    name=f"Image_{memory.id}", 
                                    path=memory.image_path, 
                                    display="inline"
                                ))
                            if memory.model_path and os.path.exists(memory.model_path):
                                elements.append(cl.File(
                                    name=f"Model_{memory.id}", 
                                    path=memory.model_path, 
                                    display="inline"
                                ))
            
            # No need for additional deduplication since we're already filtering
            response_text = conversation_tracker.format_memory_response(
                memories,
                parsed_query.get('intent', 'recall')
            )
            
            print(f"DEBUG: Final response with {len(memories)} memories and {len(elements)} elements")
            
            # Update processing message with results
            processing_msg.content = response_text
            processing_msg.elements = elements
            await processing_msg.update()
            
            # Add action to create similar content
            if memories:
                actions = [cl.Action(name="create_similar_to_memory", label="🎨 Create Similar", payload={"memory_id": memories[0].id, "prompt": memories[0].original_prompt})]
                await cl.Message(content="**Actions:**", actions=actions).send()
        
        else:
            print("DEBUG: No search results found even after fallback")
            processing_msg.content = "🤔 I couldn't find any memories. This shouldn't happen since you have memories stored. Check the console for debug info."
            await processing_msg.update()
    
    except Exception as e:
        print(f"DEBUG: Memory query failed: {e}")
        import traceback
        traceback.print_exc()
        processing_msg.content = f"❌ Memory search failed: {str(e)}\n\nDEBUG: Check console for details"
        await processing_msg.update()


@cl.action_callback("create_similar_to_memory")
async def on_create_similar_to_memory(action) -> None:
    """
    Handle creation of similar content based on existing memory.
    
    Creates a variation prompt based on an existing memory entry and
    initiates the generation process with memory context.
    """
    memory_id = action.payload["memory_id"]
    original_prompt = action.payload["prompt"]
    variation_prompt = f"Create a variation of: {original_prompt}"
    await handle_generation(variation_prompt, reference_memory_id=memory_id)


def verify_file_permissions(filepath: str) -> bool:
    """Verify file exists and is accessible."""
    try:
        if not os.path.exists(filepath):
            return False
        # Check read permissions
        with open(filepath, 'rb') as f:
            f.read(1)
        return True
    except Exception as e:
        print(f"File access error {filepath}: {e}")
        return False

# Update copy_file_safely function
def copy_file_safely(src: str, dest: str) -> Optional[str]:
    """Safely copy a file with permission checks."""
    try:
        if src and verify_file_permissions(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            # Ensure copied file is readable
            os.chmod(dest, 0o644)
            return dest if verify_file_permissions(dest) else None
    except Exception as e:
        print(f"Error copying file {src} to {dest}: {e}")
    return None


def get_proper_extension(file_path: str) -> str:
    """
    Determine the proper file extension for 3D model files.
    
    Analyzes file headers and content to determine if the file is
    GLB (binary) or GLTF (text) format for proper file handling.
    
    Args:
        file_path (str): Path to the 3D model file to analyze.
        
    Returns:
        str: Proper file extension ('.glb' or '.gltf').
    """
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


async def handle_generation(prompt: str, reference_memory_id: Optional[str] = None) -> None:
    """
    Handle 3D model generation with memory integration and file management.
    
    Processes generation requests by analyzing memory context, calling the API,
    managing generated files, and storing results in the memory system.
    
    Args:
        prompt (str): Text prompt for 3D model generation.
        reference_memory_id (Optional[str]): Optional reference to existing memory for variations.
    """
    start_time = time.time()
    user_id = cl.user_session.get("user_id", "default")
    session_id = cl.user_session.get("session_id")
    
    processing_msg = cl.Message(content="")
    await processing_msg.send()
    
    try:
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 1/3:** Analyzing memory context...
**Status:** Searching for relevant past creations"""
        await processing_msg.update()
        
        memory_context = []
        seen_prompts = set()

        # Build memory context for enhanced generation
        if reference_memory_id:
            with sqlite3.connect(memory_manager.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (reference_memory_id,))
                row = cursor.fetchone()
                if row:
                    memory = memory_manager._row_to_memory_entry(row)
                    memory_context.append(memory)
                    seen_prompts.add(memory.original_prompt.lower().strip())
        else:
            search_results = semantic_search.semantic_search(prompt, user_id, limit=3)
            for result in search_results:
                memories = memory_manager.search_memories(result.matched_content, user_id, limit=1)
                for memory in memories:
                    if memory.original_prompt.lower().strip() not in seen_prompts:
                        memory_context.append(memory)
                        seen_prompts.add(memory.original_prompt.lower().strip())
        
        # Create enhanced prompt with memory context
        enhanced_prompt = prompt
        if memory_context:
            context_prompts = [mem.original_prompt for mem in memory_context[:2]]
            enhanced_prompt = f"{prompt} (inspired by: {', '.join(context_prompts)})"
        
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 2/3:** Calling API for generation...
**Status:** This may take 1-2 minutes for high-quality results
**Memory Context:** {len(memory_context)} relevant memories found"""
        await processing_msg.update()
        
        response = await cl.make_async(call_api_sync)(enhanced_prompt)
        
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
            
            latest_image = max(image_files, key=os.path.getctime) if image_files else None
            latest_model = max(model_files, key=os.path.getctime) if model_files else None
            
            elements = []
            download_info = []
            timestamp = int(time.time())
            
            # Process image file
            image_path = None
            if latest_image and os.path.exists(latest_image):
                ext = os.path.splitext(latest_image)[1] or '.png'
                image_filename = f"image_{timestamp}{ext}"
                image_path = os.path.join(DOWNLOADS_DIR, image_filename)
                
                copied_image_path = copy_file_safely(latest_image, image_path)
                if copied_image_path and verify_file_permissions(copied_image_path):
                    print(f"DEBUG: Successfully copied image to {copied_image_path}")
                    
                    # Add image display element (consistent with memory display)
                    elements.append(
                        cl.Image(
                            name=f"Image_{timestamp}",
                            path=copied_image_path,
                            display="inline"
                        )
                    )
                    
                    # Add image download button
                    elements.append(
                        cl.File(
                            name=f"ImageDownload_{timestamp}",
                            path=copied_image_path,
                            display="inline"
                        )
                    )
                    
                    download_info.append(f"🖼️ **Image:** `{image_filename}`")
                else:
                    print(f"DEBUG: Failed to copy or verify image file: {latest_image}")
            
            # Process 3D model file
            model_path = None
            if latest_model and os.path.exists(latest_model):
                proper_ext = get_proper_extension(latest_model)
                model_filename = f"model_{timestamp}{proper_ext}"
                model_path = os.path.join(DOWNLOADS_DIR, model_filename)
                
                copied_model_path = copy_file_safely(latest_model, model_path)
                if copied_model_path and verify_file_permissions(copied_model_path):
                    print(f"DEBUG: Successfully copied model to {copied_model_path}")
                    
                    # Add model download button (consistent with memory display)
                    elements.append(
                        cl.File(
                            name=f"Model_{timestamp}",
                            path=copied_model_path,
                            display="inline"
                        )
                    )
                    
                    download_info.append(f"🗿 **3D Model:** `{model_filename}`")
                    file_size = os.path.getsize(copied_model_path)
                    download_info.append(f"   └── Size: {file_size:,} bytes")
                    download_info.append(f"   └── Format: {proper_ext.upper()}")
                else:
                    print(f"DEBUG: Failed to copy or verify model file: {latest_model}")
            
            # Store in memory system
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
                print(f"DEBUG: Stored memory with ID: {memory_id}")
            except Exception as e:
                print(f"Failed to store in memory: {e}")
                import traceback
                traceback.print_exc()
            
            total_time = int(time.time() - start_time)
            
            # Create success message
            success_content = f"""## ✅ Generation Complete! ({total_time}s)

**Prompt:** "{prompt}"
**Memory Context:** {len(memory_context)} relevant memories used

**Generated Files:**
{chr(10).join(download_info)}

**Downloads:** Click the download buttons above to save files
**Location:** `{os.path.abspath(DOWNLOADS_DIR)}`

**💾 Memory:** This creation has been stored and can be recalled later!"""

            # Update the message with content and elements
            processing_msg.content = success_content
            processing_msg.elements = elements
            await processing_msg.update()
            
            # Send additional information
            await cl.Message(content="""**3D File Usage:**
- 🔧 **Blender:** File → Import → glTF 2.0 (.glb/.gltf)
- 🎮 **Unity:** Drag file into Assets folder
- 🌐 **Online:** gltf-viewer.donmccurdy.com""").send()
            
            # Add action buttons
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
async def generate_another(action) -> None:
    """Handle generate another action callback."""
    await cl.Message(content="Great! Just type your new description and I'll generate another 3D model for you. 🎨").send()


@cl.action_callback("create_variation")
async def create_variation(action) -> None:
    """Handle create variation action callback."""
    await handle_generation(action.payload["prompt"])


@cl.action_callback("view_memories")
async def view_memories(action) -> None:
    """Handle view memories action callback."""
    await handle_memory_query(action.payload["prompt"])

# Utility function for deduplicating memory context
def deduplicate_memory_context(memories: List[MemoryEntry]) -> List[MemoryEntry]:
    """Deduplicate memory context based on content similarity."""
    seen = set()
    deduped = []
    
    for memory in memories:
        # Create a unique key combining prompt and timestamp
        key = f"{memory.original_prompt.lower().strip()}_{memory.timestamp}"
        if key not in seen:
            seen.add(key)
            deduped.append(memory)
            
    return deduped

def cleanup_files() -> None:
    """Clean up old generated files."""
    try:
        # Clean up downloads older than 7 days
        now = time.time()
        max_age = 7 * 24 * 60 * 60  # 7 days in seconds
        
        for directory in [DOWNLOADS_DIR, "outputs"]:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        if os.stat(filepath).st_mtime < now - max_age:
                            try:
                                os.remove(filepath)
                                print(f"Cleaned up old file: {filepath}")
                            except Exception as e:
                                print(f"Failed to remove {filepath}: {e}")
    except Exception as e:
        print(f"Cleanup error: {e}")

def check_environment() -> None:
    """Check and initialize required environment."""
    required_dirs = ["downloads", "outputs", "logs"]
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            # Ensure directory is readable and writable
            os.chmod(directory, 0o755)
            print(f"✓ Checked directory: {directory}")
        except Exception as e:
            print(f"! Directory error {directory}: {e}")
            raise

# Add to main
if __name__ == "__main__":
    check_environment()
    cleanup_files()
    cl.run()