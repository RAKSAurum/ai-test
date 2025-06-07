import gradio as gr
import requests
import json
import os
from PIL import Image
import logging
from pathlib import Path
import shutil
import time

class AI3DGeneratorGUI:
    def __init__(self, api_url="http://localhost:8888"):
        self.api_url = api_url
        self.latest_image_path = None
        self.latest_model_path = None
        # Create downloads directory for file access
        self.downloads_dir = "downloads"
        os.makedirs(self.downloads_dir, exist_ok=True)

    def copy_files_for_download(self, image_path, model_path):
        """Copy generated files to downloads directory for easy access"""
        download_info = []
        
        if image_path and os.path.exists(image_path):
            # Copy image to downloads
            image_name = f"generated_image_{int(time.time())}.png"
            download_image_path = os.path.join(self.downloads_dir, image_name)
            shutil.copy2(image_path, download_image_path)
            download_info.append(f"Image: {download_image_path}")
        
        if model_path and os.path.exists(model_path):
            # Copy model to downloads
            model_ext = os.path.splitext(model_path)[1]
            model_name = f"generated_model_{int(time.time())}{model_ext}"
            download_model_path = os.path.join(self.downloads_dir, model_name)
            shutil.copy2(model_path, download_model_path)
            download_info.append(f"3D Model: {download_model_path}")
        
        return download_info

    def generate_3d_model(self, prompt, progress=gr.Progress()):
        """Call your AI pipeline through the API with progress tracking"""
        try:
            progress(0.1, desc="Starting generation...")
            
            # Call your API
            response = requests.post(
                f"{self.api_url}/execution",
                json={"prompt": prompt, "attachments": []},
                timeout=120
            )
            
            progress(0.9, desc="Processing results...")
            
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
                
                progress(1.0, desc="Complete!")
                
                # Get latest files
                latest_image = None
                latest_model = None
                
                if image_files:
                    latest_image = max(image_files, key=os.path.getctime)
                    self.latest_image_path = latest_image
                
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    self.latest_model_path = latest_model
                
                # Copy files for download
                download_info = self.copy_files_for_download(latest_image, latest_model)
                download_text = "\n".join(download_info) if download_info else "No files generated"
                
                # Return results
                return (
                    latest_image,
                    latest_model if latest_model and latest_model.endswith('.glb') else None,
                    f"✓ {message}\n\nFiles saved to: downloads/\n{download_text}",
                    gr.update(visible=bool(latest_image)),
                    gr.update(visible=bool(latest_model))
                )
            else:
                return (
                    None,
                    None,
                    f"Error: HTTP {response.status_code}",
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

    def clear_outputs(self):
        """Clear all outputs"""
        self.latest_image_path = None
        self.latest_model_path = None
        return (
            "",  # prompt
            None,  # generated_image
            None,  # model_3d
            "",  # status_text
            gr.update(visible=False),  # image_section
            gr.update(visible=False)   # model_section
        )

    def create_interface(self):
        """Create the minimalistic Gradio interface"""
        with gr.Blocks(
            title="AI 3D Generator",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter")
            ),
            css="""
            .gradio-container {
                max-width: 900px !important;
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
                min-height: 100px;
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
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>AI 3D Generator</h1>
                <p>Transform ideas into 3D models</p>
            </div>
            """)
            
            # Input Section
            with gr.Group(elem_classes=["input-section"]):
                prompt_input = gr.Textbox(
                    label="Describe your 3D model",
                    placeholder="A futuristic robot with glowing blue eyes and metallic armor...",
                    lines=3,
                    show_label=False
                )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate",
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
            
            # Quick Examples
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["Futuristic robot with glowing blue eyes"],
                        ["Medieval castle on floating island"],
                        ["Steampunk airship with brass details"],
                        ["Cyberpunk motorcycle with neon lights"]
                    ],
                    inputs=prompt_input,
                    label="Quick Examples"
                )
            
            # Output Section
            with gr.Group(elem_classes=["output-section"]):
                status_text = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False,
                    elem_classes=["status-box"],
                    show_label=False,
                    placeholder="Click Generate to start..."
                )
                
                # Results Grid
                with gr.Row(elem_classes=["results-grid"]):
                    # Image preview
                    with gr.Group(visible=False, elem_classes=["preview-container"]) as image_section:
                        generated_image = gr.Image(
                            label="Generated Image",
                            type="filepath",
                            interactive=False,
                            show_label=True,
                            height=300
                        )
                    
                    # 3D model preview
                    with gr.Group(visible=False, elem_classes=["preview-container"]) as model_section:
                        model_3d = gr.Model3D(
                            label="3D Model",
                            clear_color=[0.95, 0.95, 0.97, 1.0],
                            height=300
                        )
            
            # Download Info
            gr.HTML(f"""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f1f5f9; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <strong>📁 Downloads:</strong> <code>{os.path.abspath(self.downloads_dir)}</code>
                <br><small style="color: #64748b;">Generated files are automatically saved here</small>
            </div>
            """)
            
            # Event handlers
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
            
            interface.queue()
            return interface

def main():
    """Launch the Gradio interface"""
    logging.basicConfig(level=logging.INFO)
    
    # Create GUI instance
    gui = AI3DGeneratorGUI()
    
    # Create and launch interface
    interface = gui.create_interface()
    
    print("🚀 Starting AI 3D Model Generator...")
    print(f"📁 Downloads: {os.path.abspath(gui.downloads_dir)}")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()