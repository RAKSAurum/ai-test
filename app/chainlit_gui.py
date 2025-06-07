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
    """Initialize chat with welcome message and quick actions"""
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
    
    # Add quick action buttons
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
    await handle_generation(action.payload["prompt"])

@cl.action_callback("castle")
async def on_action_castle(action):
    await handle_generation(action.payload["prompt"])

@cl.action_callback("dragon")
async def on_action_dragon(action):
    await handle_generation(action.payload["prompt"])

@cl.on_message
async def main(message: cl.Message):
    await handle_generation(message.content)

async def handle_generation(prompt: str):
    """Core generation workflow with real-time updates"""
    start_time = time.time()
    
    # Initialize processing message
    processing_msg = cl.Message(content="")
    await processing_msg.send()
    
    try:
        # Phase 1: Image Generation
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 1/2:** Creating image via queue workflow...
**Status:** Submitting to Openfabric AI queue"""
        await processing_msg.update()
        
        # Call your working pipeline with async wrapper
        results = await cl.make_async(generate_image_and_convert_to_3d)(prompt, "both")
        
        # Phase 2: 3D Conversion
        processing_msg.content = f"""## 🎯 Generating: {prompt}

**Step 2/2:** Converting to 3D model...
**Status:** Processing with direct execution workflow"""
        await processing_msg.update()
        
        # Prepare files for display
        elements = []
        download_info = []
        
        # Copy files to downloads with timestamps
        def copy_to_downloads(src, prefix):
            if src and os.path.exists(src):
                ext = os.path.splitext(src)[1]
                dest = f"{DOWNLOADS_DIR}/{prefix}_{int(time.time())}{ext}"
                shutil.copy2(src, dest)
                return dest
            return None
        
        # Handle image
        if 'original_image' in results:
            image_path = copy_to_downloads(results['original_image'], "image")
            if image_path:
                elements.append(cl.Image(
                    name="Generated Image", 
                    path=image_path, 
                    display="inline",
                    size="large"
                ))
                download_info.append(f"🖼️ **Image:** `{os.path.basename(image_path)}`")
        
        # Handle 3D model
        if 'object' in results:
            model_path = copy_to_downloads(results['object'], "model")
            if model_path:
                elements.append(cl.File(
                    name="3D Model", 
                    path=model_path
                ))
                download_info.append(f"🗿 **3D Model:** `{os.path.basename(model_path)}`")
        
        # Handle video
        if 'video' in results:
            video_path = copy_to_downloads(results['video'], "video")
            if video_path:
                elements.append(cl.File(
                    name="Rotation Video", 
                    path=video_path
                ))
                download_info.append(f"🎥 **Video:** `{os.path.basename(video_path)}`")
        
        # Final success message
        total_time = int(time.time() - start_time)
        success_content = f"""## ✅ Generation Complete! ({total_time}s)

**Prompt:** "{prompt}"

**Generated Files:**
{chr(10).join(download_info)}

**Downloads Location:** `{os.path.abspath(DOWNLOADS_DIR)}`

Your files are ready for download and use in Blender, Unity, or any 3D software!"""
        
        await cl.Message(
            content=success_content,
            elements=elements
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
- "A blue sphere"""
        await processing_msg.update()

@cl.action_callback("generate_another")
async def generate_another(action):
    await cl.Message(
        content="Great! Just type your new description and I'll generate another 3D model for you. 🎨"
    ).send()

if __name__ == "__main__":
    cl.run()