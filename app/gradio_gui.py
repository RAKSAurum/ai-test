"""
AI 3D Generator Gradio Interface with Memory Integration

A Gradio-based web interface that provides AI-powered 3D model generation
with intelligent memory management and natural language query capabilities.
"""

import os
import sys
import time
import logging
import gradio as gr

import json
import shutil
import sqlite3

from pathlib import Path
from typing import Optional, List

import requests
from PIL import Image

# Add path for memory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import memory system components
from memory import MemoryManager, ConversationTracker, SemanticSearch
from memory.semantic_search import SearchResult

class AI3DGeneratorGUI:
    """
    Gradio-based GUI for AI 3D model generation with integrated memory system.
    
    This class provides a web interface for generating 3D models from text prompts
    while maintaining a comprehensive memory system for storing, searching, and
    recalling previous generations using natural language queries.
    """

    def __init__(self, api_url: str = "http://localhost:8888") -> None:
        """
        Initialize the AI 3D Generator GUI with memory system integration.
        
        Args:
            api_url (str): URL of the AI generation API. Defaults to "http://localhost:8888".
        """
        self.api_url = api_url
        self.latest_image_path: Optional[str] = None
        self.latest_model_path: Optional[str] = None
        
        # Create downloads directory for file access
        self.downloads_dir = "downloads"
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Initialize memory system (completely local)
        self.memory_manager = MemoryManager()
        self.conversation_tracker = ConversationTracker()
        self.semantic_search = SemanticSearch()
        
        # Session management (local only) - Use consistent user ID like Chainlit
        self.current_user_id = "default"  # Changed to match Chainlit
        self.current_session_id = self.memory_manager.create_session(self.current_user_id)
        
        logging.info("🧠 Memory system initialized for Gradio GUI (local only)")

    def handle_memory_query(self, query: str, progress=gr.Progress()) -> tuple:
        """
        Handle memory-related queries with comprehensive search and display.
        Enhanced to match Chainlit's robust memory handling logic exactly.
        """
        try:
            progress(0.1, desc="🧠 Searching through your memories...")
            print(f"DEBUG: Memory query from user {self.current_user_id}: '{query}'")
            
            # Parse the memory query
            parsed_query = self.conversation_tracker.parse_memory_query(query)
            print(f"DEBUG: Parsed query result: {parsed_query}")
            
            if not parsed_query['is_memory_query']:
                debug_info = f"""Not recognized as memory query.
                
DEBUG INFO:
- Query: '{query}'
- Parsed: {parsed_query}
- Try: 'show me my recent creations' or 'list my recent work'"""
                return (
                    None,
                    None,
                    debug_info,
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            search_query = self.conversation_tracker.build_search_query(parsed_query)
            print(f"DEBUG: Search query: {search_query}")
            
            progress(0.3, desc="Executing search strategy...")
            
            # Determine search strategy based on intent (exact Chainlit logic)
            if parsed_query['intent'] in ['recall', 'list']:
                print("DEBUG: Using recent memories for recall/list intent")
                recent_memories = self.memory_manager.get_recent_memories(self.current_user_id, limit=10)
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
                search_results = self.semantic_search.temporal_search(
                    self.current_user_id, search_query['time_range'], limit=10
                )
            else:
                print("DEBUG: Using hybrid search")
                search_results = self.semantic_search.hybrid_search(
                    search_query['text_search'], self.current_user_id, 
                    search_query.get('time_range'), 
                    search_query.get('entity_filters'), limit=10
                )
                
                # If hybrid search returns nothing, fall back to recent memories
                if not search_results:
                    print("DEBUG: Hybrid search returned 0 results, falling back to recent memories")
                    recent_memories = self.memory_manager.get_recent_memories(self.current_user_id, limit=5)
                    search_results = [
                        SearchResult(
                            memory_id=mem.id,
                            relevance_score=1.0,
                            match_type='recent_fallback',
                            matched_content=mem.original_prompt
                        ) for mem in recent_memories
                    ]
            
            print(f"DEBUG: Found {len(search_results)} initial results")
            
            progress(0.7, desc="Processing search results...")
            
            # Get full memory details and create response (exact Chainlit approach)
            if search_results:
                memories = []
                seen_prompts = set()  # Track unique prompts
                
                for result in search_results:
                    with sqlite3.connect(self.memory_manager.db_path) as conn:
                        cursor = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (result.memory_id,))
                        row = cursor.fetchone()
                        if row:
                            memory = self.memory_manager._row_to_memory_entry(row)
                            # Only add if we haven't seen this prompt before
                            if memory.original_prompt.lower().strip() not in seen_prompts:
                                memories.append(memory)
                                seen_prompts.add(memory.original_prompt.lower().strip())
                
                # Format response using conversation tracker
                response_text = self.conversation_tracker.format_memory_response(
                    memories,
                    parsed_query.get('intent', 'recall')
                )
                
                print(f"DEBUG: Final response with {len(memories)} memories")
                
                # Get the most recent memory for display
                if memories:
                    latest_memory = memories[0]
                    display_image = latest_memory.image_path if latest_memory.image_path and os.path.exists(latest_memory.image_path) else None
                    display_model = latest_memory.model_path if latest_memory.model_path and os.path.exists(latest_memory.model_path) else None
                    
                    progress(1.0, desc="Memory search complete!")
                    
                    return (
                        display_image,
                        display_model,
                        response_text,
                        gr.update(visible=bool(display_image)),
                        gr.update(visible=bool(display_model))
                    )
                else:
                    return (
                        None,
                        None,
                        "🤔 Found search results but couldn't load memory details. Check console for debug info.",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            
            else:
                print("DEBUG: No search results found even after fallback")
                return (
                    None,
                    None,
                    "🤔 I couldn't find any memories. This shouldn't happen since you have memories stored. Check the console for debug info.",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
        except Exception as e:
            print(f"DEBUG: Memory query failed: {e}")
            import traceback
            traceback.print_exc()
            return (
                None,
                None,
                f"❌ Memory search failed: {str(e)}\n\nDEBUG: Check console for details",
                gr.update(visible=False),
                gr.update(visible=False)
            )

    def handle_generation(self, prompt: str, reference_memory_id: Optional[str] = None, progress=gr.Progress()) -> tuple:
        """
        Handle 3D model generation with memory integration and file management.
        Enhanced to match Chainlit's exact generation logic.
        """
        start_time = time.time()
        
        try:
            progress(0.1, desc=f"🎯 Generating: {prompt}")
            
            # Build memory context for enhanced generation (exact Chainlit logic)
            memory_context = []
            seen_prompts = set()
            
            if reference_memory_id:
                with sqlite3.connect(self.memory_manager.db_path) as conn:
                    cursor = conn.execute("SELECT * FROM memory_entries WHERE id = ?", (reference_memory_id,))
                    row = cursor.fetchone()
                    if row:
                        memory = self.memory_manager._row_to_memory_entry(row)
                        memory_context.append(memory)
                        seen_prompts.add(memory.original_prompt.lower().strip())
            else:
                search_results = self.semantic_search.semantic_search(prompt, self.current_user_id, limit=3)
                for result in search_results:
                    memories = self.memory_manager.search_memories(result.matched_content, self.current_user_id, limit=1)
                    for memory in memories:
                        if memory.original_prompt.lower().strip() not in seen_prompts:
                            memory_context.append(memory)
                            seen_prompts.add(memory.original_prompt.lower().strip())
            
            # Create enhanced prompt with memory context
            enhanced_prompt = prompt
            if memory_context:
                context_prompts = [mem.original_prompt for mem in memory_context[:2]]
                enhanced_prompt = f"{prompt} (inspired by: {', '.join(context_prompts)})"
            
            progress(0.3, desc="Calling API for generation...")
            
            response = requests.post(
                f"{self.api_url}/execution",
                json={"prompt": enhanced_prompt},
                timeout=300
            )
            
            progress(0.6, desc="Processing results and storing in memory...")
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "Generation completed")
                
                # Find generated files (exact Chainlit logic)
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
                
                timestamp = int(time.time())
                download_info = []
                
                # Process image file (exact Chainlit logic)
                image_path = None
                if latest_image and os.path.exists(latest_image):
                    ext = os.path.splitext(latest_image)[1] or '.png'
                    image_filename = f"image_{timestamp}{ext}"
                    image_path = os.path.join(self.downloads_dir, image_filename)
                    
                    copied_image_path = self.copy_file_safely(latest_image, image_path)
                    if copied_image_path and self.verify_file_permissions(copied_image_path):
                        print(f"DEBUG: Successfully copied image to {copied_image_path}")
                        download_info.append(f"🖼️ **Image:** `{image_filename}`")
                    else:
                        print(f"DEBUG: Failed to copy or verify image file: {latest_image}")
                
                # Process 3D model file (exact Chainlit logic)
                model_path = None
                if latest_model and os.path.exists(latest_model):
                    proper_ext = self.get_proper_extension(latest_model)
                    model_filename = f"model_{timestamp}{proper_ext}"
                    model_path = os.path.join(self.downloads_dir, model_filename)
                    
                    copied_model_path = self.copy_file_safely(latest_model, model_path)
                    if copied_model_path and self.verify_file_permissions(copied_model_path):
                        print(f"DEBUG: Successfully copied model to {copied_model_path}")
                        file_size = os.path.getsize(copied_model_path)
                        download_info.append(f"🗿 **3D Model:** `{model_filename}`")
                        download_info.append(f"   └── Size: {file_size:,} bytes")
                        download_info.append(f"   └── Format: {proper_ext.upper()}")
                    else:
                        print(f"DEBUG: Failed to copy or verify model file: {latest_model}")
                
                # Store in memory system (exact Chainlit logic)
                try:
                    memory_metadata = {
                        'generation_time': time.time(),
                        'processing_time': time.time() - start_time,
                        'reference_memory_id': reference_memory_id,
                        'memory_context_used': len(memory_context) > 0,
                        'api_response': message
                    }
                    
                    memory_id = self.memory_manager.store_generation(
                        session_id=self.current_session_id,
                        user_id=self.current_user_id,
                        original_prompt=prompt,
                        enhanced_prompt=enhanced_prompt,
                        image_path=image_path,
                        model_path=model_path,
                        metadata=memory_metadata
                    )
                    
                    self.semantic_search.add_memory_embedding(memory_id, f"{prompt} {enhanced_prompt}")
                    print(f"DEBUG: Stored memory with ID: {memory_id}")
                except Exception as e:
                    print(f"Failed to store in memory: {e}")
                    import traceback
                    traceback.print_exc()
                
                total_time = int(time.time() - start_time)
                
                # Create success message (exact Chainlit format)
                success_content = f"""## ✅ Generation Complete! ({total_time}s)

**Prompt:** "{prompt}"
**Memory Context:** {len(memory_context)} relevant memories used

**Generated Files:**
{chr(10).join(download_info)}

**Downloads:** Files saved to downloads directory
**Location:** `{os.path.abspath(self.downloads_dir)}`

**💾 Memory:** This creation has been stored and can be recalled later!"""

                progress(1.0, desc="Generation complete!")
                
                return (
                    image_path,
                    model_path,
                    success_content,
                    gr.update(visible=bool(image_path)),
                    gr.update(visible=bool(model_path))
                )
                
            else:
                error_text = f"""## ❌ Generation Failed

**Error:** HTTP {response.status_code}
**Response:** {response.text[:200]}

**Your API is running but returned an error. Check the server logs.**"""
                
                return (None, None, error_text, gr.update(visible=False), gr.update(visible=False))
                
        except requests.exceptions.ConnectionError:
            error_text = f"""## ❌ Connection Failed

**Error:** Cannot connect to API server at {self.api_url}

**Please check:**
1. Is your API server running?
2. Try: `python main.py` to start the server"""
            
            return (None, None, error_text, gr.update(visible=False), gr.update(visible=False))
            
        except Exception as e:
            error_text = f"""## ❌ Generation Failed

**Error:** {str(e)}

**Memory Commands still work:**
- "Show me my recent creations"
- "Find my robot from yesterday\""""
            
            return (None, None, error_text, gr.update(visible=False), gr.update(visible=False))

    def process_input(self, prompt: str, progress=gr.Progress()) -> tuple:
        """
        Main input handler that routes to generation or memory query.
        Enhanced to match Chainlit's exact routing logic.
        """
        # Parse input to determine if it's a memory query (exact Chainlit logic)
        parsed_query = self.conversation_tracker.parse_memory_query(prompt)
        print(f"DEBUG: Input '{prompt}' parsed as: {parsed_query}")
        
        if parsed_query['is_memory_query'] and parsed_query['intent'] in ['recall', 'list']:
            return self.handle_memory_query(prompt, progress)
        else:
            return self.handle_generation(prompt, None, progress)

    def get_proper_extension(self, file_path: str) -> str:
        """
        Determine the proper file extension for 3D model files.
        Exact copy from Chainlit implementation.
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

    def verify_file_permissions(self, filepath: str) -> bool:
        """Verify file exists and is accessible. Exact copy from Chainlit."""
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

    def copy_file_safely(self, src: str, dest: str) -> Optional[str]:
        """Safely copy a file with permission checks. Exact copy from Chainlit."""
        try:
            if src and self.verify_file_permissions(src):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                # Ensure copied file is readable
                os.chmod(dest, 0o644)
                return dest if self.verify_file_permissions(dest) else None
        except Exception as e:
            print(f"Error copying file {src} to {dest}: {e}")
        return None

    def get_recent_memories(self) -> str:
        """
        Get recent memories for display in the interface.
        Enhanced with better error handling and debug info.
        """
        try:
            recent_memories = self.memory_manager.get_recent_memories(
                self.current_user_id, limit=5
            )
            
            if recent_memories:
                memory_text = "**Your Recent Creations:**\n\n"
                for i, memory in enumerate(recent_memories, 1):
                    timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(memory.timestamp))
                    memory_text += f"{i}. {memory.original_prompt} ({timestamp})\n"
                
                memory_text += f"\n**Total memories for user {self.current_user_id}: {len(recent_memories)}**"
                return memory_text
            else:
                return "No recent memories found. Create something to get started!"
                
        except Exception as e:
            print(f"Error loading memories: {e}")
            import traceback
            traceback.print_exc()
            return f"Error loading memories: {str(e)}"

    def create_similar_to_last(self) -> str:
        """
        Create a variation prompt based on the last creation.
        """
        try:
            recent_memories = self.memory_manager.get_recent_memories(
                self.current_user_id, limit=1
            )
            
            if recent_memories:
                last_prompt = recent_memories[0].original_prompt
                variation_prompt = f"Create a variation of: {last_prompt}"
                return variation_prompt
            else:
                return "No previous creations found to base a variation on."
                
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_outputs(self) -> tuple:
        """
        Clear all outputs and reset the interface state.
        """
        self.latest_image_path = None
        self.latest_model_path = None
        return (
            "",  # prompt
            None,  # generated_image
            None,  # model_3d
            "🚀 Ready! Generate 3D models or search your memories using natural language.\n\n💡 Try: 'show me what I created yesterday' or 'create a robot'",  # status_text
            gr.update(visible=False),  # image_section
            gr.update(visible=False)   # model_section
        )

    def create_interface(self) -> gr.Blocks:
        """
        Create the comprehensive Gradio interface with memory integration.
        Enhanced to match Chainlit functionality exactly.
        """
        with gr.Blocks(
            title="AI 3D Generator with Memory",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter")
            ),
            css="""
            .gradio-container {
                max-width: 1000px !important;
                margin: 0 auto;
                font-family: 'Inter', sans-serif;
            }
            .main-header {
                text-align: center;
                margin: 2rem 0;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                color: white;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            .main-header h1 {
                margin: 0 0 0.5rem 0;
                font-size: 2.5rem;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            .main-header p {
                margin: 0;
                opacity: 0.9;
                font-size: 1.1rem;
                font-weight: 300;
            }
            .input-section {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                margin-bottom: 2rem;
            }
            .output-section {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                margin-bottom: 2rem;
            }
            .memory-section {
                background: #f8fafc;
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #3b82f6;
                margin-bottom: 2rem;
            }
            .generate-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border: none !important;
                padding: 1rem 2rem !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3) !important;
            }
            .generate-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
            }
            .memory-btn {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
            }
            .clear-btn {
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                color: #64748b !important;
                font-weight: 500 !important;
            }
            .status-box {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1rem;
                font-family: 'SF Mono', monospace;
                font-size: 0.9rem;
                color: #475569;
                min-height: 120px;
            }
            .results-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin-top: 2rem;
            }
            @media (max-width: 768px) {
                .results-grid {
                    grid-template-columns: 1fr;
                }
                .main-header h1 {
                    font-size: 2rem;
                }
            }
            .preview-container {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            }
            """
        ) as interface:
            
            # Header section with branding
            gr.HTML("""
            <div class="main-header">
                <h1>🧠 AI 3D Generator with Memory</h1>
                <p>Transform ideas into 3D models • Remember everything • Recall naturally</p>
            </div>
            """)
            
            # Memory section for displaying recent creations
            with gr.Group(elem_classes=["memory-section"]):
                gr.HTML("<h3>💾 Memory System (Local)</h3>")
                memory_display = gr.Textbox(
                    label="Recent Memories",
                    value=self.get_recent_memories(),
                    lines=4,
                    interactive=False,
                    show_label=False
                )
                
                with gr.Row():
                    refresh_memories_btn = gr.Button(
                        "🔄 Refresh Memories",
                        variant="secondary",
                        scale=2
                    )
                    create_similar_btn = gr.Button(
                        "🎨 Create Similar to Last",
                        variant="secondary",
                        elem_classes=["memory-btn"],
                        scale=2
                    )
            
            # Input section for prompts and commands
            with gr.Group(elem_classes=["input-section"]):
                prompt_input = gr.Textbox(
                    label="Describe your 3D model or use memory commands",
                    placeholder="A futuristic robot with glowing blue eyes... OR 'show me my robot from yesterday'",
                    lines=3,
                    show_label=False
                )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate / Search",
                        variant="primary",
                        elem_classes=["generate-btn"],
                        scale=4
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_classes=["clear-btn"],
                        scale=1
                    )
            
            # Quick examples with both generation and memory commands
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["Futuristic robot with glowing blue eyes"],
                        ["Medieval castle on floating island"],
                        ["show me my recent creations"],
                        ["find my robot from yesterday"],
                        ["create something like my last dragon"],
                        ["list my work from this week"]
                    ],
                    inputs=prompt_input,
                    label="Quick Examples (Generation + Memory)"
                )
            
            # Output section for results and status
            with gr.Group(elem_classes=["output-section"]):
                status_text = gr.Textbox(
                    label="Status",
                    lines=6,
                    interactive=False,
                    elem_classes=["status-box"],
                    show_label=False,
                    value="🚀 Ready! Generate 3D models or search your memories using natural language.\n\n💡 Try: 'show me what I created yesterday' or 'create a robot'"
                )
                
                # Results grid for image and 3D model display
                with gr.Row(elem_classes=["results-grid"]):
                    # Image preview section
                    with gr.Group(visible=False, elem_classes=["preview-container"]) as image_section:
                        generated_image = gr.Image(
                            label="Generated Image / Memory Image",
                            type="filepath",
                            interactive=False,
                            show_label=True,
                            height=300
                        )
                    
                    # 3D model preview section
                    with gr.Group(visible=False, elem_classes=["preview-container"]) as model_section:
                        model_3d = gr.Model3D(
                            label="3D Model / Memory Model",
                            clear_color=[0.95, 0.95, 0.97, 1.0],
                            height=300
                        )
            
            # Download and memory location information
            gr.HTML(f"""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f1f5f9; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <strong>📁 Downloads:</strong> <code>{os.path.abspath(self.downloads_dir)}</code>
                <br><strong>🧠 Memory Database:</strong> <code>{os.path.abspath('memory/ai_memory.db')}</code>
                <br><small style="color: #64748b;">Generated files and memories are automatically saved • API schema unchanged</small>
            </div>
            """)
            
            # Event handlers for user interactions
            generate_btn.click(
                fn=self.process_input,  # Changed to use unified input handler
                inputs=[prompt_input],
                outputs=[
                    generated_image,
                    model_3d,
                    status_text,
                    image_section,
                    model_section
                ],
                show_progress=True
            )
            
            clear_btn.click(
                fn=self.clear_outputs,
                outputs=[
                    prompt_input,
                    generated_image,
                    model_3d,
                    status_text,
                    image_section,
                    model_section
                ]
            )
            
            refresh_memories_btn.click(
                fn=self.get_recent_memories,
                outputs=[memory_display]
            )
            
            create_similar_btn.click(
                fn=self.create_similar_to_last,
                outputs=[prompt_input]
            )
            
            interface.queue()
            return interface

    def cleanup_old_files(self, max_age_days: int = 7) -> None:
        """Remove files older than max_age_days."""
        try:
            now = time.time()
            for directory in [self.downloads_dir, "outputs"]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath):
                            if os.stat(filepath).st_mtime < now - max_age_days * 86400:
                                try:
                                    os.remove(filepath)
                                    logging.info(f"Removed old file: {filepath}")
                                except Exception as e:
                                    logging.error(f"Failed to remove {filepath}: {e}")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")


def main() -> None:
    """
    Launch the Gradio interface with comprehensive memory system integration.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Create GUI instance
    gui = AI3DGeneratorGUI()
    gui.cleanup_old_files()
    
    # Create and launch interface
    interface = gui.create_interface()
    
    print("🚀 Starting AI 3D Model Generator with Memory...")
    print(f"📁 Downloads: {os.path.abspath(gui.downloads_dir)}")
    print(f"🧠 Memory Database: {os.path.abspath('memory/ai_memory.db')}")
    print(f"👤 User Session: {gui.current_session_id}")
    print("🔧 API Schema: Unchanged - only sending 'prompt' field")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()