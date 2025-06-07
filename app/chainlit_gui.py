import chainlit as cl
import os
import time
import shutil
from image_to_3d_pipeline import generate_image_and_convert_to_3d

# Configuration
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

@cl.on_chat_start
async def start():
    """
    Initialize chat with welcome message and quick action examples.
    
    Displays a welcome message explaining the AI 3D Generator workflow
    and provides quick action buttons for common generation examples.
    """
    await cl.Message(
        content="""# 🚀 AI 3D Generator
**Transform text into 3D models using your proven pipeline**

**How it works:**
1. 🧠 Enhanced prompts using DeepSeek LLM
2. 🎨 Queue-based image generation via Openfabric
3. 🗿 Direct 3D model conversion with GLB output
4. 📥 Automatic file downloads

Just describe what you want to create!"""
    ).send()
    
    # Add quick action buttons for common examples
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
        )
    ]
    
    await cl.Message(
        content="**Quick Examples:**",
        actions=actions
    ).send()

@cl.action_callback("robot")
async def on_action_robot(action):
    """Handle robot example action callback."""
    await handle_generation(action.payload["prompt"])

@cl.action_callback("castle")
async def on_action_castle(action):
    """Handle castle example action callback."""
    await handle_generation(action.payload["prompt"])

@cl.action_callback("dragon")
async def on_action_dragon(action):
    """Handle dragon example action callback."""
    await handle_generation(action.payload["prompt"])

@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler for user input.
    
    Args:
        message (cl.Message): The user's message containing the generation prompt
    """
    await handle_generation(message.content)

def copy_file_safely(src, dest):
    """
    Safely copy file with proper binary handling for GLB files.
    
    Args:
        src (str): Source file path
        dest (str): Destination file path
        
    Returns:
        str or None: Destination path if successful, None if failed
    """
    try:
        if src and os.path.exists(src):
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            # Copy file in binary mode to preserve GLB structure
            with open(src, 'rb') as src_file:
                with open(dest, 'wb') as dest_file:
                    shutil.copyfileobj(src_file, dest_file)
            
            return dest
    except Exception as e:
        print(f"Error copying file {src} to {dest}: {e}")
    return None

def get_proper_extension(file_path):
    """
    Determine the proper file extension based on file content.
    Enhanced version that handles GLTF/GLB detection more reliably.
    
    Args:
        file_path (str): Path to the file to check
        
    Returns:
        str: Proper file extension (.glb or .gltf)
    """
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to determine format
            header = f.read(12)
            
            # GLB files start with 'glTF' magic number (0x46546C67)
            if header[:4] == b'glTF':
                return '.glb'
            else:
                # Check if it's a JSON-based GLTF file
                f.seek(0)
                try:
                    content = f.read(500).decode('utf-8', errors='ignore')
                    if '"asset"' in content and '"version"' in content and '"generator"' in content:
                        return '.gltf'
                except:
                    pass
                
                # Default to GLB for binary 3D content
                return '.glb'
    except Exception:
        return '.gltf'  # Safe fallback

async def handle_generation(prompt: str):
    """
    Core generation workflow with real-time updates and proper file handling.
    
    Handles the complete pipeline from text prompt to 3D model generation,
    including image creation, 3D conversion, file management with correct extensions,
    and user feedback. Fixed to ensure proper filename handling for downloads.
    
    Args:
        prompt (str): User's text description for the 3D model to generate
    """
    start_time = time.time()
    
    # Initialize processing message for real-time updates
    processing_msg = cl.Message(content="")
    await processing_msg.send()
    
    try:
        # Phase 1: Image Generation
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 1/2:** Creating image via queue workflow...
**Status:** Submitting to Openfabric AI queue"""
        await processing_msg.update()
        
        # Call the main pipeline with async wrapper
        results = await cl.make_async(generate_image_and_convert_to_3d)(prompt, "both")
        
        # Phase 2: 3D Conversion
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 2/2:** Converting to 3D model...
**Status:** Processing with direct execution workflow"""
        await processing_msg.update()
        
        # Prepare files for display and download
        elements = []
        download_info = []
        
        # Create timestamped filename base for unique file naming
        timestamp = int(time.time())
        
        # Handle generated image
        if 'original_image' in results and results['original_image']:
            src_image = results['original_image']
            if os.path.exists(src_image):
                # Preserve original file extension
                ext = os.path.splitext(src_image)[1] or '.png'
                image_filename = f"image_{timestamp}{ext}"
                image_dest = os.path.join(DOWNLOADS_DIR, image_filename)
                
                if copy_file_safely(src_image, image_dest):
                    elements.append(cl.Image(
                        name=image_filename,  # Explicit filename with extension
                        path=image_dest, 
                        display="inline",
                        size="large"
                    ))
                    download_info.append(f"🖼️ **Image:** `{image_filename}`")
        
        # Handle 3D model with proper extension detection and filename handling
        if 'object' in results and results['object']:
            src_model = results['object']
            if os.path.exists(src_model):
                # Detect proper extension based on file content
                proper_ext = get_proper_extension(src_model)
                model_filename = f"model_{timestamp}{proper_ext}"
                model_dest = os.path.join(DOWNLOADS_DIR, model_filename)
                
                if copy_file_safely(src_model, model_dest):
                    # Set appropriate MIME type based on extension
                    mime_type = "model/gltf-binary" if proper_ext == '.glb' else "model/gltf+json"
                    
                    # Create file element with explicit filename to preserve extension
                    elements.append(cl.File(
                        name=model_filename,  # Explicit filename with extension
                        path=model_dest,
                        display="inline",
                        mime=mime_type
                    ))
                    download_info.append(f"🗿 **3D Model:** `{model_filename}`")
                    
                    # Add file size and format information
                    file_size = os.path.getsize(model_dest)
                    download_info.append(f"   └── Size: {file_size:,} bytes")
                    download_info.append(f"   └── Format: {proper_ext.upper()} ({mime_type})")
        
        # Handle rotation video (if generated)
        if 'video' in results and results['video']:
            src_video = results['video']
            if os.path.exists(src_video):
                ext = os.path.splitext(src_video)[1] or '.mp4'
                video_filename = f"video_{timestamp}{ext}"
                video_dest = os.path.join(DOWNLOADS_DIR, video_filename)
                
                if copy_file_safely(src_video, video_dest):
                    elements.append(cl.File(
                        name=video_filename,  # Explicit filename with extension
                        path=video_dest,
                        display="inline"
                    ))
                    download_info.append(f"🎥 **Video:** `{video_filename}`")
        
        # Display final success message with generation results
        total_time = int(time.time() - start_time)
        success_content = f"""## ✅ Generation Complete! ({total_time}s)

**Prompt:** "{prompt}"

**Generated Files:**
{chr(10).join(download_info)}

**Downloads Location:** `{os.path.abspath(DOWNLOADS_DIR)}`

**Note:** Files are automatically saved with correct extensions. Click the attachments below to download with proper filenames."""
        
        await cl.Message(
            content=success_content,
            elements=elements
        ).send()
        
        # Add usage tips for 3D files with Linux-specific advice
        await cl.Message(
            content="""**3D File Usage Tips:**
- 📁 Files download with correct extensions (.glb or .gltf) automatically
- 🔧 **Blender:** File → Import → glTF 2.0 (.glb/.gltf)
- 🎮 **Unity:** Drag the file directly into Assets folder
- 🌐 **Online Viewers:** threejs.org/editor or gltf-viewer.donmccurdy.com
- 📱 **Mobile:** Use apps like "3D Model Viewer" or "GLB Viewer"

**Linux Fix:** If downloaded file has no extension, rename it to add `.glb` or `.gltf`"""
        ).send()
        
        # Add action for next generation
        actions = [
            cl.Action(
                name="generate_another",
                label="🎨 Generate Another",
                payload={"action": "new"}
            )
        ]
        
        await cl.Message(
            content="**What's next?**",
            actions=actions
        ).send()
        
    except Exception as e:
        # Handle errors with helpful troubleshooting information
        processing_msg.content = f"""## ❌ Generation Failed

**Error:** {str(e)}

**Troubleshooting:**
1. Check your internet connection
2. Verify Openfabric APIs are accessible
3. Try simpler prompts like "red cube" or "wooden chair"
4. Contact support if issue persists

**Example simple prompts:**
- "A simple red cube"
- "A wooden chair"
- "A blue sphere"
- "A small house"
- "A cartoon car\""""
        await processing_msg.update()

@cl.action_callback("generate_another")
async def generate_another(action):
    """Handle generate another action callback."""
    await cl.Message(
        content="Great! Just type your new description and I'll generate another 3D model for you. 🎨"
    ).send()

if __name__ == "__main__":
    cl.run()