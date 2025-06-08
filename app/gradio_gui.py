"""
AI 3D Generator Gradio Interface with Memory Integration

A Gradio-based web interface that provides AI-powered 3D model generation
with intelligent memory management and natural language query capabilities.
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import requests
from PIL import Image

# Add path for memory imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import memory system components
from memory import MemoryManager, ConversationTracker, SemanticSearch


class AI3DGeneratorGUI:
    """
    Gradio-based GUI for AI 3D model generation with integrated memory system.
    
    This class provides a web interface for generating 3D models from text prompts
    while maintaining a comprehensive memory system for storing, searching, and
    recalling previous generations using natural language queries.
    
    The interface supports both direct generation requests and memory-based queries,
    allowing users to create variations of previous work and search through their
    creation history using conversational language.
    
    Attributes:
        api_url (str): URL of the AI generation API endpoint.
        latest_image_path (Optional[str]): Path to the most recently generated image.
        latest_model_path (Optional[str]): Path to the most recently generated 3D model.
        downloads_dir (str): Directory for storing downloadable files.
        memory_manager (MemoryManager): Core memory management system.
        conversation_tracker (ConversationTracker): Natural language processing system.
        semantic_search (SemanticSearch): Semantic search and retrieval system.
        current_user_id (str): Identifier for the current user session.
        current_session_id (str): Identifier for the current session.
    
    Example:
        >>> gui = AI3DGeneratorGUI("http://localhost:8888")
        >>> interface = gui.create_interface()
        >>> interface.launch()
    """

    def __init__(self, api_url: str = "http://localhost:8888") -> None:
        """
        Initialize the AI 3D Generator GUI with memory system integration.
        
        Sets up the API connection, file management directories, and initializes
        the complete memory system for local storage and retrieval.
        
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
        
        # Session management (local only)
        self.current_user_id = "gradio_user"
        self.current_session_id = self.memory_manager.create_session(self.current_user_id)
        
        logging.info("🧠 Memory system initialized for Gradio GUI (local only)")

    def copy_files_for_download(self, image_path: Optional[str], model_path: Optional[str]) -> list:
        """
        Copy generated files to downloads directory for easy user access.
        
        Creates timestamped copies of generated files in the downloads directory
        to provide users with easily accessible files for download and use.
        
        Args:
            image_path (Optional[str]): Path to the generated image file.
            model_path (Optional[str]): Path to the generated 3D model file.
            
        Returns:
            list: List of download information strings describing copied files.
        """
        download_info = []
        
        if image_path and os.path.exists(image_path):
            # Copy image to downloads with timestamp
            image_name = f"generated_image_{int(time.time())}.png"
            download_image_path = os.path.join(self.downloads_dir, image_name)
            shutil.copy2(image_path, download_image_path)
            download_info.append(f"Image: {download_image_path}")
        
        if model_path and os.path.exists(model_path):
            # Copy model to downloads with timestamp and original extension
            model_ext = os.path.splitext(model_path)[1]
            model_name = f"generated_model_{int(time.time())}{model_ext}"
            download_model_path = os.path.join(self.downloads_dir, model_name)
            shutil.copy2(model_path, download_model_path)
            download_info.append(f"3D Model: {download_model_path}")
        
        return download_info

    def search_memories(self, query: str, progress=gr.Progress()) -> tuple:
        """
        Search through memory using natural language queries with progress tracking.
        
        Processes natural language memory queries to find relevant past generations,
        supporting temporal queries, entity-based searches, and general recall requests.
        
        Args:
            query (str): Natural language query for searching memories.
            progress (gr.Progress): Gradio progress tracker for user feedback.
            
        Returns:
            tuple: Contains (image_path, model_path, response_text, image_visibility, model_visibility)
                for displaying search results in the Gradio interface.
        """
        try:
            progress(0.2, desc="Parsing memory query...")
            
            # Parse the memory query using conversation tracker
            parsed_query = self.conversation_tracker.parse_memory_query(query)
            
            if not parsed_query['is_memory_query']:
                return (
                    None,
                    None,
                    "This doesn't appear to be a memory query. Try phrases like:\n"
                    "- 'Show me what I created yesterday'\n"
                    "- 'Find my robot creations'\n"
                    "- 'List my recent work'",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            progress(0.5, desc="Searching memories...")
            
            # Build search parameters from parsed query
            search_query = self.conversation_tracker.build_search_query(parsed_query)
            
            # Perform search based on query type and available parameters
            if search_query['time_range']:
                # Temporal search for time-based queries
                search_results = self.semantic_search.temporal_search(
                    self.current_user_id, search_query['time_range'], limit=10
                )
            elif search_query['entity_filters'] or search_query['text_search']:
                # Hybrid search for entity or text-based queries
                search_results = self.semantic_search.hybrid_search(
                    search_query['text_search'],
                    self.current_user_id,
                    search_query.get('time_range'),
                    search_query.get('entity_filters'),
                    limit=10
                )
            else:
                # Recent memories fallback for general queries
                recent_memories = self.memory_manager.get_recent_memories(
                    self.current_user_id, limit=5
                )
                search_results = [
                    self.semantic_search.SearchResult(
                        memory_id=mem.id,
                        relevance_score=1.0,
                        match_type='recent',
                        matched_content=mem.original_prompt
                    ) for mem in recent_memories
                ]
            
            progress(0.8, desc="Formatting results...")
            
            if search_results:
                # Get full memory details from search results
                memories = []
                for result in search_results:
                    with self.memory_manager._get_connection() as conn:
                        cursor = conn.execute("""
                            SELECT * FROM memory_entries WHERE id = ?
                        """, (result.memory_id,))
                        row = cursor.fetchone()
                        if row:
                            memory = self.memory_manager._row_to_memory_entry(row)
                            memories.append(memory)
                
                # Format response using conversation tracker
                response_text = self.conversation_tracker.format_memory_response(
                    memories, parsed_query.get('intent', 'recall')
                )
                
                # Get the most recent memory for display
                if memories:
                    latest_memory = memories[0]
                    display_image = latest_memory.image_path if latest_memory.image_path and os.path.exists(latest_memory.image_path) else None
                    display_model = latest_memory.model_path if latest_memory.model_path and os.path.exists(latest_memory.model_path) else None
                    
                    progress(1.0, desc="Complete!")
                    
                    return (
                        display_image,
                        display_model,
                        response_text,
                        gr.update(visible=bool(display_image)),
                        gr.update(visible=bool(display_model))
                    )
            
            progress(1.0, desc="No results found")
            
            return (
                None,
                None,
                "🤔 I couldn't find any memories matching your request. Try describing what you're looking for differently.",
                gr.update(visible=False),
                gr.update(visible=False)
            )
            
        except Exception as e:
            return (
                None,
                None,
                f"❌ Memory search failed: {str(e)}",
                gr.update(visible=False),
                gr.update(visible=False)
            )

    def generate_3d_model(self, prompt: str, progress=gr.Progress()) -> tuple:
        """
        Generate 3D model through API with memory integration and progress tracking.
        
        Handles both generation requests and memory queries, enhances prompts using
        memory context, calls the AI API, and stores results in the memory system.
        
        Args:
            prompt (str): User input prompt for generation or memory query.
            progress (gr.Progress): Gradio progress tracker for user feedback.
            
        Returns:
            tuple: Contains (image_path, model_path, status_message, image_visibility, model_visibility)
                for displaying results in the Gradio interface.
        """
        try:
            progress(0.1, desc="Starting generation...")
            
            # Check if this is a memory query first (handle locally)
            parsed_query = self.conversation_tracker.parse_memory_query(prompt)
            if parsed_query['is_memory_query'] and parsed_query['intent'] in ['recall', 'list']:
                return self.search_memories(prompt, progress)
            
            progress(0.2, desc="Getting memory context...")
            
            # Get memory context for prompt enhancement (local only)
            memory_context = []
            enhanced_prompt = prompt  # Default to original prompt
            
            if parsed_query['intent'] == 'create_similar':
                # Search for similar past creations for context
                search_results = self.semantic_search.semantic_search(
                    prompt, self.current_user_id, limit=3
                )
                for result in search_results:
                    memories = self.memory_manager.search_memories(
                        result.matched_content, self.current_user_id, limit=1
                    )
                    memory_context.extend(memories)
                
                # Enhance prompt locally using memory context
                if memory_context:
                    context_prompts = [mem.original_prompt for mem in memory_context[:2]]
                    enhanced_prompt = f"{prompt} (inspired by: {', '.join(context_prompts)})"
            
            progress(0.3, desc="Calling AI pipeline...")
            
            # Use exact original schema - only send prompt
            api_payload = {
                "prompt": enhanced_prompt  # Use enhanced prompt but keep original API schema
            }
            
            # Call API with exact original format
            response = requests.post(
                f"{self.api_url}/execution",
                json=api_payload,
                timeout=120
            )
            
            progress(0.8, desc="Processing results...")
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "Generation completed")
                
                # Find generated files in outputs directory
                image_files = []
                model_files = []
                
                if os.path.exists("outputs"):
                    for file in os.listdir("outputs"):
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(f"outputs/{file}")
                        elif file.endswith(('.glb', '.obj', '.ply', '.stl')):
                            model_files.append(f"outputs/{file}")
                
                progress(0.9, desc="Storing in memory...")
                
                # Get latest files based on creation time
                latest_image = None
                latest_model = None
                
                if image_files:
                    latest_image = max(image_files, key=os.path.getctime)
                    self.latest_image_path = latest_image
                
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    self.latest_model_path = latest_model
                
                # Store generation in memory (completely local)
                try:
                    memory_metadata = {
                        'generation_time': time.time(),
                        'api_response': message,
                        'memory_context_used': len(memory_context) > 0,
                        'interface': 'gradio',
                        'enhanced_prompt': enhanced_prompt,
                        'original_prompt': prompt
                    }
                    
                    memory_id = self.memory_manager.store_generation(
                        session_id=self.current_session_id,
                        user_id=self.current_user_id,
                        original_prompt=prompt,
                        enhanced_prompt=enhanced_prompt,
                        image_path=latest_image,
                        model_path=latest_model,
                        metadata=memory_metadata
                    )
                    
                    # Add to semantic search index for future searches
                    self.semantic_search.add_memory_embedding(
                        memory_id, f"{prompt} {enhanced_prompt}"
                    )
                    
                    logging.info(f"💾 Stored generation in memory: {memory_id}")
                    
                except Exception as e:
                    logging.error(f"Failed to store in memory: {e}")
                
                # Copy files for download
                download_info = self.copy_files_for_download(latest_image, latest_model)
                download_text = "\n".join(download_info) if download_info else "No files generated"
                
                progress(1.0, desc="Complete!")
                
                # Enhanced status message with memory info
                status_message = f"""✓ {message}

💾 **Memory:** Generation stored for future recall
🧠 **Context:** {len(memory_context)} relevant memories used
📁 **Files:** {download_text}

**Memory Commands:**
- "Show me what I created yesterday"
- "Find my robot creations"  
- "Create something like my last dragon"
- "List my recent work\""""
                
                # Return results for Gradio interface
                return (
                    latest_image,
                    latest_model if latest_model and latest_model.endswith('.glb') else None,
                    status_message,
                    gr.update(visible=bool(latest_image)),
                    gr.update(visible=bool(latest_model))
                )
            else:
                return (
                    None,
                    None,
                    f"Error: HTTP {response.status_code}\nResponse: {response.text[:200]}",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
                
        except requests.exceptions.Timeout:
            return (
                None,
                None,
                "Error: Request timed out (>2 minutes)",
                gr.update(visible=False),
                gr.update(visible=False)
            )
        except Exception as e:
            return (
                None,
                None,
                f"Error: {str(e)}",
                gr.update(visible=False),
                gr.update(visible=False)
            )

    def get_recent_memories(self) -> str:
        """
        Get recent memories for display in the interface.
        
        Retrieves and formats recent memory entries for display in the
        memory section of the Gradio interface.
        
        Returns:
            str: Formatted string containing recent memory information.
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
                
                return memory_text
            else:
                return "No recent memories found. Create something to get started!"
                
        except Exception as e:
            return f"Error loading memories: {str(e)}"

    def create_similar_to_last(self) -> str:
        """
        Create a variation prompt based on the last creation.
        
        Retrieves the most recent creation and generates a variation prompt
        for creating similar content with modifications.
        
        Returns:
            str: Variation prompt based on the last creation, or error message.
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
        
        Resets all interface elements to their initial state and clears
        stored file paths for a fresh start.
        
        Returns:
            tuple: Reset values for all interface elements.
        """
        self.latest_image_path = None
        self.latest_model_path = None
        return (
            "",  # prompt
            None,  # generated_image
            None,  # model_3d
            "Click Generate to start or use memory commands like 'show me my recent creations'",  # status_text
            gr.update(visible=False),  # image_section
            gr.update(visible=False)   # model_section
        )

    def create_interface(self) -> gr.Blocks:
        """
        Create the comprehensive Gradio interface with memory integration.
        
        Builds a modern, responsive web interface with integrated memory system,
        supporting both 3D model generation and natural language memory queries.
        
        Returns:
            gr.Blocks: Configured Gradio interface ready for launch.
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
                fn=self.generate_3d_model,
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


def main() -> None:
    """
    Launch the Gradio interface with comprehensive memory system integration.
    
    Initializes the GUI, creates the interface, and launches the web server
    with detailed logging and configuration information.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Create GUI instance
    gui = AI3DGeneratorGUI()
    
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