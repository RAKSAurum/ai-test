import gradio as gr
import requests
import json
import os
from PIL import Image
import logging

class AI3DGeneratorGUI:
    def __init__(self, api_url="http://localhost:8888"):
        self.api_url = api_url
        
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
                
                # Try to find generated files
                image_files = []
                model_files = []
                
                if os.path.exists("outputs"):
                    for file in os.listdir("outputs"):
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(f"outputs/{file}")
                        elif file.endswith(('.obj', '.ply', '.stl')):
                            model_files.append(f"outputs/{file}")
                
                progress(1.0, desc="Complete!")
                
                # Return the latest generated image if available
                latest_image = None
                if image_files:
                    latest_image = max(image_files, key=os.path.getctime)
                
                return message, latest_image, f"Generated files: {len(image_files)} images, {len(model_files)} models"
                
            else:
                return f"Error: HTTP {response.status_code}", None, "Generation failed"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out (>2 minutes)", None, "Try a simpler prompt"
        except Exception as e:
            return f"Error: {str(e)}", None, "Check if the API server is running"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="🤖 AI 3D Model Generator",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🤖 AI 3D Model Generator</h1>
                <p>Transform your ideas into 3D models using DeepSeek LLM and Openfabric AI</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    gr.Markdown("## 📝 Describe Your 3D Model")
                    
                    prompt_input = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="A steampunk mechanical dragon with brass gears and copper wings...",
                        lines=4,
                        max_lines=8
                    )
                    
                    with gr.Row():
                        generate_btn = gr.Button(
                            "🎨 Generate 3D Model", 
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["A futuristic robot with glowing blue eyes and metallic armor"],
                            ["A medieval castle on a floating island with waterfalls"],
                            ["A steampunk airship with copper details and brass propellers"],
                            ["A mystical forest creature with crystalline antlers"],
                            ["A cyberpunk motorcycle with neon underglow"],
                            ["An ancient temple with intricate stone carvings"]
                        ],
                        inputs=prompt_input,
                        label="💡 Example Prompts"
                    )
                
                with gr.Column(scale=2):
                    # Output section
                    gr.Markdown("## 🎯 Generation Results")
                    
                    result_text = gr.Textbox(
                        label="Generation Status",
                        lines=8,
                        max_lines=12,
                        interactive=False
                    )
                    
                    generated_image = gr.Image(
                        label="Generated Image Preview",
                        type="filepath",
                        interactive=False
                    )
                    
                    file_info = gr.Textbox(
                        label="Generated Files",
                        interactive=False
                    )
            
            # Info section
            with gr.Row():
                gr.Markdown("""
                ## ℹ️ How It Works
                
                1. **🧠 LLM Enhancement**: Your prompt is enhanced using DeepSeek LLM for better artistic details
                2. **🖼️ Image Generation**: Enhanced prompt is converted to a high-quality image
                3. **🗿 3D Model Creation**: Image is transformed into a downloadable 3D model
                4. **💾 Memory Storage**: Generation is saved for future reference and context
                
                **💡 Tips for Better Results:**
                - Be descriptive about materials, colors, and style
                - Mention lighting preferences (dramatic, soft, cinematic)
                - Include artistic styles (realistic, cartoon, steampunk, etc.)
                - Specify details like texture and finish
                """)
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_3d_model,
                inputs=[prompt_input],
                outputs=[result_text, generated_image, file_info],
                show_progress=True
            )
            
            clear_btn.click(
                fn=lambda: ("", None, ""),
                outputs=[prompt_input, generated_image, file_info]
            )
        
        return interface

def main():
    """Launch the Gradio interface"""
    logging.basicConfig(level=logging.INFO)
    
    # Create GUI instance
    gui = AI3DGeneratorGUI()
    
    # Create and launch interface
    interface = gui.create_interface()
    
    print("🚀 Starting AI 3D Model Generator GUI...")
    print("📊 API Server should be running on http://localhost:8888")
    print("🌐 GUI will be available on http://localhost:7860")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()