import gradio as gr
import requests
import json
import os
from PIL import Image
import logging
from pathlib import Path
import shutil
import time
from datetime import datetime

class ModernAI3DGeneratorGUI:
    def __init__(self, api_url="http://localhost:8888"):
        self.api_url = api_url
        self.latest_image_path = None
        self.latest_model_path = None
        self.downloads_dir = "downloads"
        self.generation_history = []
        os.makedirs(self.downloads_dir, exist_ok=True)

    def copy_files_for_download(self, image_path, model_path):
        """Copy generated files to downloads directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_info = []
        
        if image_path and os.path.exists(image_path):
            image_name = f"ai_image_{timestamp}.png"
            download_image_path = os.path.join(self.downloads_dir, image_name)
            shutil.copy2(image_path, download_image_path)
            download_info.append(f"🖼️ {image_name}")
        
        if model_path and os.path.exists(model_path):
            model_ext = os.path.splitext(model_path)[1]
            model_name = f"ai_model_{timestamp}{model_ext}"
            download_model_path = os.path.join(self.downloads_dir, model_name)
            shutil.copy2(model_path, download_model_path)
            download_info.append(f"🗿 {model_name}")
        
        return download_info

    def generate_3d_model(self, prompt, progress=gr.Progress()):
        """Enhanced generation with modern progress tracking"""
        if not prompt.strip():
            return (
                None, None, "⚠️ Please enter a description", 
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), ""
            )
        
        try:
            # Enhanced progress stages
            progress(0.05, desc="🚀 Initializing...")
            time.sleep(0.5)
            
            progress(0.15, desc="🧠 Processing prompt...")
            response = requests.post(
                f"{self.api_url}/execution",
                json={"prompt": prompt, "attachments": []},
                timeout=120
            )
            
            progress(0.50, desc="🎨 Generating image...")
            time.sleep(1)
            
            progress(0.80, desc="🗿 Creating 3D model...")
            time.sleep(1)
            
            progress(0.95, desc="💾 Saving files...")
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "Generation completed successfully!")
                
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
                
                self.latest_image_path = latest_image
                self.latest_model_path = latest_model
                
                # Copy files and update history
                download_info = self.copy_files_for_download(latest_image, latest_model)
                
                # Add to history
                generation_data = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "files": len(download_info),
                    "status": "Success"
                }
                self.generation_history.append(generation_data)
                if len(self.generation_history) > 5:
                    self.generation_history.pop(0)
                
                progress(1.0, desc="✅ Complete!")
                
                # Create status message
                status_msg = f"✅ **Generation Complete!**\n\n📝 **Prompt:** {prompt}\n\n"
                if download_info:
                    status_msg += f"📁 **Files Created:**\n" + "\n".join([f"   • {info}" for info in download_info])
                    status_msg += f"\n\n💾 **Location:** `{self.downloads_dir}/`"
                else:
                    status_msg += "⚠️ No files were generated"
                
                # History display
                history_display = self._format_history()
                
                return (
                    latest_image,
                    latest_model if latest_model and latest_model.endswith('.glb') else None,
                    status_msg,
                    gr.update(visible=bool(latest_image)),
                    gr.update(visible=bool(latest_model)),
                    gr.update(visible=True),
                    history_display
                )
            else:
                error_msg = f"❌ **Generation Failed**\n\nHTTP Error {response.status_code}\nPlease check if the API server is running."
                return (None, None, error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")
                
        except requests.exceptions.Timeout:
            error_msg = "⏱️ **Request Timeout**\n\nThe generation took longer than 2 minutes.\nTry a simpler prompt or check your connection."
            return (None, None, error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")
        except Exception as e:
            error_msg = f"🚨 **Unexpected Error**\n\n{str(e)}\n\nPlease check if the API server is running on {self.api_url}"
            return (None, None, error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")

    def _format_history(self):
        """Format generation history for display"""
        if not self.generation_history:
            return "No generations yet"
        
        history_text = "**Recent Generations:**\n\n"
        for i, gen in enumerate(reversed(self.generation_history[-3:]), 1):
            status_icon = "✅" if gen["status"] == "Success" else "❌"
            history_text += f"{status_icon} **{gen['timestamp']}** - {gen['prompt']} ({gen['files']} files)\n"
        
        return history_text

    def clear_all(self):
        """Clear all outputs and reset interface"""
        self.latest_image_path = None
        self.latest_model_path = None
        return (
            "",  # prompt
            None,  # image
            None,  # model
            "🎯 **Ready to Generate**\n\nEnter your description above and click Generate to create amazing 3D models!",  # status
            gr.update(visible=False),  # image section
            gr.update(visible=False),  # model section
            gr.update(visible=False),  # download section
            ""  # history
        )

    def get_download_info(self):
        """Get formatted download information"""
        if not os.path.exists(self.downloads_dir):
            return "📁 No files available"
        
        files = [f for f in os.listdir(self.downloads_dir) if not f.startswith('.')]
        if not files:
            return "📁 No files available"
        
        # Sort by creation time, newest first
        files.sort(key=lambda x: os.path.getctime(os.path.join(self.downloads_dir, x)), reverse=True)
        
        info_text = f"📁 **Download Folder:** `{os.path.abspath(self.downloads_dir)}`\n\n"
        info_text += f"**Recent Files ({len(files[:5])} of {len(files)}):**\n"
        
        for file in files[:5]:
            file_path = os.path.join(self.downloads_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%H:%M")
            
            icon = "🖼️" if file.endswith(('.png', '.jpg', '.jpeg')) else "🗿"
            info_text += f"{icon} `{file}` ({file_size:.1f}KB) - {mod_time}\n"
        
        return info_text

    def create_interface(self):
        """Create the modern, premium interface"""
        
        # Custom theme with modern styling
        theme = gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue", 
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ).set(
            body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            body_background_fill_dark="linear-gradient(135deg, #2D1B69 0%, #11998e 100%)"
        )

        with gr.Blocks(
            title="AI 3D Generator Pro",
            theme=theme,
            css="""
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Global Styles */
            .gradio-container {
                max-width: 1400px !important;
                margin: 0 auto !important;
                background: transparent !important;
            }
            
            /* Glassmorphism Effect */
            .glass-card {
                background: rgba(255, 255, 255, 0.1) !important;
                backdrop-filter: blur(20px) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 24px !important;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
                padding: 2rem !important;
                margin: 1rem 0 !important;
            }
            
            .glass-card-dark {
                background: rgba(0, 0, 0, 0.3) !important;
                backdrop-filter: blur(20px) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 20px !important;
                padding: 1.5rem !important;
            }
            
            /* Header Styling */
            .hero-header {
                text-align: center;
                padding: 4rem 2rem;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(30px);
                border-radius: 32px;
                margin: 2rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            }
            
            .hero-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%),
                           linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%);
                background-size: 60px 60px;
                animation: float 20s infinite linear;
                opacity: 0.1;
            }
            
            @keyframes float {
                0% { transform: translateX(0px) translateY(0px); }
                100% { transform: translateX(-60px) translateY(-60px); }
            }
            
            .hero-title {
                font-size: 4rem !important;
                font-weight: 700 !important;
                background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                margin: 0 0 1rem 0 !important;
                letter-spacing: -0.02em;
                position: relative;
                z-index: 2;
            }
            
            .hero-subtitle {
                font-size: 1.3rem !important;
                color: rgba(255, 255, 255, 0.8) !important;
                font-weight: 300 !important;
                margin: 0 !important;
                position: relative;
                z-index: 2;
            }
            
            /* Input Styling */
            .prompt-input textarea {
                background: rgba(255, 255, 255, 0.9) !important;
                border: 2px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 16px !important;
                padding: 1.5rem !important;
                font-size: 1.1rem !important;
                min-height: 120px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
            }
            
            .prompt-input textarea:focus {
                border-color: #8b5cf6 !important;
                box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1) !important;
                transform: translateY(-2px) !important;
            }
            
            .prompt-input label {
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                color: white !important;
                margin-bottom: 1rem !important;
            }
            
            /* Button Styling */
            .generate-btn {
                background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%) !important;
                border: none !important;
                padding: 1.2rem 3rem !important;
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                border-radius: 16px !important;
                color: white !important;
                transition: all 0.4s ease !important;
                box-shadow: 0 12px 24px rgba(139, 92, 246, 0.3) !important;
                position: relative !important;
                overflow: hidden !important;
            }
            
            .generate-btn:hover {
                transform: translateY(-4px) scale(1.02) !important;
                box-shadow: 0 20px 40px rgba(139, 92, 246, 0.4) !important;
            }
            
            .generate-btn:active {
                transform: translateY(-2px) scale(0.98) !important;
            }
            
            .clear-btn {
                background: rgba(255, 255, 255, 0.1) !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                color: white !important;
                font-weight: 500 !important;
                border-radius: 12px !important;
                transition: all 0.3s ease !important;
                backdrop-filter: blur(10px) !important;
            }
            
            .clear-btn:hover {
                background: rgba(255, 255, 255, 0.2) !important;
                transform: translateY(-2px) !important;
            }
            
            /* Results Grid */
            .results-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin-top: 2rem;
            }
            
            @media (max-width: 1024px) {
                .results-container {
                    grid-template-columns: 1fr;
                }
                .hero-title {
                    font-size: 3rem !important;
                }
            }
            
            /* Preview Cards */
            .preview-card {
                background: rgba(255, 255, 255, 0.95) !important;
                border-radius: 20px !important;
                padding: 1.5rem !important;
                box-shadow: 0 16px 32px rgba(0,0,0,0.1) !important;
                transition: all 0.3s ease !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
            }
            
            .preview-card:hover {
                transform: translateY(-4px) !important;
                box-shadow: 0 24px 48px rgba(0,0,0,0.15) !important;
            }
            
            .preview-card h3 {
                margin: 0 0 1rem 0 !important;
                font-size: 1.3rem !important;
                font-weight: 600 !important;
                color: #1e293b !important;
            }
            
            /* Status Display */
            .status-display {
                background: rgba(255, 255, 255, 0.95) !important;
                border: none !important;
                border-radius: 16px !important;
                padding: 2rem !important;
                font-size: 1rem !important;
                line-height: 1.6 !important;
                box-shadow: 0 12px 24px rgba(0,0,0,0.08) !important;
                min-height: 200px !important;
            }
            
            /* Download Section */
            .download-info {
                background: rgba(6, 182, 212, 0.1) !important;
                border: 2px solid rgba(6, 182, 212, 0.3) !important;
                border-radius: 16px !important;
                padding: 2rem !important;
                margin: 2rem 0 !important;
                backdrop-filter: blur(10px) !important;
            }
            
            .download-info h3 {
                color: white !important;
                margin: 0 0 1rem 0 !important;
                font-size: 1.4rem !important;
                font-weight: 600 !important;
            }
            
            .download-path {
                background: rgba(0, 0, 0, 0.2) !important;
                padding: 1rem !important;
                border-radius: 8px !important;
                font-family: 'SF Mono', monospace !important;
                color: #e2e8f0 !important;
                margin: 1rem 0 !important;
                word-break: break-all;
            }
            
            /* Examples Grid */
            .examples-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin: 2rem 0;
            }
            
            .example-card {
                background: rgba(255, 255, 255, 0.1) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 12px !important;
                padding: 1rem !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                backdrop-filter: blur(10px) !important;
            }
            
            .example-card:hover {
                background: rgba(255, 255, 255, 0.2) !important;
                transform: translateY(-2px) !important;
            }
            
            /* Animations */
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .generating {
                animation: pulse 2s infinite;
            }
            
            /* Mobile Responsiveness */
            @media (max-width: 768px) {
                .glass-card {
                    margin: 0.5rem !important;
                    padding: 1rem !important;
                }
                
                .hero-header {
                    padding: 2rem 1rem !important;
                    margin: 1rem 0 !important;
                }
                
                .generate-btn {
                    padding: 1rem 2rem !important;
                    font-size: 1rem !important;
                }
            }
            """
        ) as interface:
            
            # Hero Header
            gr.HTML("""
            <div class="hero-header">
                <h1 class="hero-title">AI 3D Generator</h1>
                <p class="hero-subtitle">Transform your imagination into stunning 3D models with cutting-edge AI</p>
            </div>
            """)
            
            # Main Input Section
            with gr.Group(elem_classes=["glass-card"]):
                prompt_input = gr.Textbox(
                    label="✨ Describe your vision",
                    placeholder="A majestic dragon perched on a crystal mountain, with iridescent scales catching moonlight and ancient runes glowing along its wings...",
                    lines=4,
                    elem_classes=["prompt-input"]
                )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate 3D Model",
                        variant="primary",
                        elem_classes=["generate-btn"],
                        scale=3
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        elem_classes=["clear-btn"],
                        scale=1
                    )
            
            # Quick Examples
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("### 💡 **Inspiration Gallery**", elem_classes=["section-title"])
                gr.Examples(
                    examples=[
                        ["Cyberpunk samurai warrior with neon katana and LED armor"],
                        ["Floating steampunk city with brass gears and copper pipes"],
                        ["Crystal cave with luminescent mushrooms and flowing water"],
                        ["Space station with rotating rings and solar panels"],
                        ["Medieval dragon made of living stone and moss"],
                        ["Futuristic hover car with holographic displays"]
                    ],
                    inputs=prompt_input,
                    elem_classes=["examples-grid"]
                )
            
            # Status and Results Section
            with gr.Group(elem_classes=["glass-card"]):
                status_text = gr.Textbox(
                    value="🎯 **Ready to Generate**\n\nEnter your description above and click Generate to create amazing 3D models!",
                    label="Generation Status",
                    lines=8,
                    interactive=False,
                    elem_classes=["status-display"]
                )
                
                # Results Grid
                with gr.Row(elem_classes=["results-container"]):
                    # Image Preview
                    with gr.Group(visible=False, elem_classes=["preview-card"]) as image_section:
                        gr.HTML("<h3>🖼️ Generated Image</h3>")
                        generated_image = gr.Image(
                            type="filepath",
                            interactive=False,
                            show_label=False,
                            height=400,
                            show_download_button=True
                        )
                    
                    # 3D Model Preview  
                    with gr.Group(visible=False, elem_classes=["preview-card"]) as model_section:
                        gr.HTML("<h3>🗿 3D Model</h3>")
                        model_3d = gr.Model3D(
                            clear_color=[0.1, 0.1, 0.15, 1.0],
                            height=400,
                            show_label=False
                        )
            
            # Download Information
            with gr.Group(visible=False, elem_classes=["download-info"]) as download_section:
                gr.HTML("<h3>📁 Download Center</h3>")
                download_path = gr.HTML(f'<div class="download-path">📂 {os.path.abspath(self.downloads_dir)}</div>')
                history_display = gr.Textbox(
                    label="Generation History",
                    lines=4,
                    interactive=False,
                    show_label=False
                )
            
            # Performance Stats (Hidden until generation)
            with gr.Group(elem_classes=["glass-card-dark"], visible=False) as stats_section:
                gr.HTML("""
                <div style="text-align: center; color: rgba(255,255,255,0.8);">
                    <h4 style="margin: 0 0 1rem 0; color: white;">🚀 System Status</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;">
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 600;">⚡</div>
                            <div>Fast Generation</div>
                        </div>
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 600;">🎯</div>
                            <div>High Quality</div>
                        </div>
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 600;">🔄</div>
                            <div>Always Learning</div>
                        </div>
                    </div>
                </div>
                """)
            
            # Event Handlers
            generate_btn.click(
                fn=self.generate_3d_model,
                inputs=[prompt_input],
                outputs=[
                    generated_image,
                    model_3d, 
                    status_text,
                    image_section,
                    model_section,
                    download_section,
                    history_display
                ],
                show_progress=True
            )
            
            clear_btn.click(
                fn=self.clear_all,
                outputs=[
                    prompt_input,
                    generated_image,
                    model_3d,
                    status_text,
                    image_section,
                    model_section,
                    download_section,
                    history_display
                ]
            )
            
            # Auto-refresh download info when section becomes visible
            download_section.change(
                fn=self.get_download_info,
                outputs=[history_display]
            )
            
            interface.queue(max_size=10, concurrency_count=2)
            return interface

def main():
    """Launch the modern interface"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create GUI instance
    gui = ModernAI3DGeneratorGUI()
    
    # Create and launch interface
    interface = gui.create_interface()
    
    print("🚀 AI 3D Generator Pro - Starting...")
    print("🌐 Interface: http://localhost:7860")
    print(f"📁 Downloads: {os.path.abspath(gui.downloads_dir)}")
    print("✨ Features: Glassmorphism UI, Progress Tracking, File Management")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        favicon_path=None,
        show_tips=True
    )

if __name__ == "__main__":
    main()