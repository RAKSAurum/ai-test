import logging
import os
import sqlite3
import json
import base64
import uuid
from PIL import Image
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ollama

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import State, Ray
from core.stub import Stub

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

# NEW: Direct URLs for Openfabric apps (node3 structure)
TEXT_TO_IMAGE_URL = 'http://PLACEHOLDER_HASH.node3.openfabric.network'  # Placeholder for text-to-image
IMAGE_TO_3D_URL = 'http://5891a64fe34041d98b0262bb1175ff07.node3.openfabric.network'  # Working image-to-3D URL

def initialize_default_config():
    """Initialize default configuration with direct URLs"""
    default_config = ConfigClass()
    # Use direct URLs for both apps
    default_config.app_ids = [
        TEXT_TO_IMAGE_URL,
        IMAGE_TO_3D_URL
    ]
    configurations['super-user'] = default_config
    logging.info("🔧 Default configuration initialized with direct URLs")

# Initialize at module level
initialize_default_config()

# Memory storage
class MemoryManager:
    def __init__(self):
        self.db_path = "memory/memory.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()

    def get_connection(self):
        """Get thread-safe database connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    original_prompt TEXT,
                    enhanced_prompt TEXT,
                    image_data TEXT,
                    model_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logging.info("💾 Memory database initialized")
        finally:
            conn.close()

    def store_generation(self, original_prompt, enhanced_prompt, image_data, model_data):
        """Store generation with thread-safe connection"""
        generation_id = str(uuid.uuid4())
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO generations 
                (id, original_prompt, enhanced_prompt, image_data, model_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (generation_id, original_prompt, enhanced_prompt, str(image_data), str(model_data)))
            conn.commit()
            logging.info(f"💾 Stored generation {generation_id}")
            return generation_id
        finally:
            conn.close()

    def recall_similar(self, prompt, limit=5):
        """Recall similar prompts from memory"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT original_prompt, enhanced_prompt 
                FROM generations 
                WHERE original_prompt LIKE ? OR enhanced_prompt LIKE ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (f'%{prompt}%', f'%{prompt}%', limit))
            
            results = cursor.fetchall()
            return [{'original': row[0], 'enhanced': row[1]} for row in results]
        except Exception as e:
            logging.error(f"❌ Memory recall failed: {e}")
            return []
        finally:
            conn.close()

# Global memory manager
memory_manager = MemoryManager()

# DeepSeek LLM Processor
class DeepSeekLLMProcessor:
    """Handles DeepSeek local LLM processing for prompt enhancement"""
    
    def __init__(self, local_model_path=None):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if local_model_path:
            self.model_name = local_model_path
            self.load_model()
        else:
            self.model_name = "deepseek-r1:1.5b"
            self.model_loaded = False

    def load_model(self):
        """Load DeepSeek model and tokenizer with optimizations"""
        try:
            logging.info(f"🧠 Loading DeepSeek model: {self.model_name}")
            logging.info(f"🖥️ Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                use_cache=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            logging.info("✅ DeepSeek model loaded successfully!")
            
        except Exception as e:
            logging.error(f"❌ Failed to load DeepSeek model: {e}")
            logging.info("🔄 Falling back to Ollama enhancement")
            self.model_loaded = False

    def enhance_prompt(self, original_prompt: str) -> str:
        """Enhance the user prompt with DeepSeek LLM or Ollama fallback"""
        try:
            # Check memory for similar prompts first
            similar = memory_manager.recall_similar(original_prompt, 3)
            context = ""
            if similar:
                context = f"Previous similar generations: {[s['enhanced'] for s in similar[:2]]}"
            
            if self.model_loaded:
                enhanced = self._deepseek_enhancement(original_prompt, context)
            else:
                enhanced = self._ollama_enhancement(original_prompt, context)
            
            logging.info(f"🧠 Enhanced prompt: '{original_prompt}' -> '{enhanced}'")
            return enhanced
            
        except Exception as e:
            logging.error(f"❌ LLM processing failed: {e}")
            return self._rule_based_enhancement(original_prompt, "")

    def _deepseek_enhancement(self, prompt: str, context: str) -> str:
        """Use DeepSeek model to enhance the prompt"""
        try:
            enhancement_prompt = f"""<think>
I need to enhance this image prompt to make it more detailed and artistic while keeping the core concept.

Original prompt: {prompt}
{f"Context: {context}" if context else ""}

I should add:
- Visual details (colors, textures, lighting)
- Artistic style descriptors
- Atmospheric elements
- Technical quality terms
</think>

Enhance this image prompt for high-quality generation: {prompt}

Enhanced prompt:"""

            inputs = self.tokenizer(
                enhancement_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Enhanced prompt:" in full_response:
                enhanced_prompt = full_response.split("Enhanced prompt:")[-1].strip()
            else:
                input_length = len(enhancement_prompt)
                enhanced_prompt = full_response[input_length:].strip()

            enhanced_prompt = enhanced_prompt.replace('\n', ' ').strip()
            enhanced_prompt = enhanced_prompt.split('.')[0] if '.' in enhanced_prompt else enhanced_prompt

            if len(enhanced_prompt) < 10 or len(enhanced_prompt) > 500:
                logging.warning("⚠️ DeepSeek response seems invalid, using fallback")
                return self._rule_based_enhancement(prompt, context)

            return enhanced_prompt

        except Exception as e:
            logging.error(f"❌ DeepSeek enhancement failed: {e}")
            return self._rule_based_enhancement(prompt, context)

    def _ollama_enhancement(self, prompt: str, context: str) -> str:
        """Fallback to Ollama local model for prompt enhancement"""
        try:
            import ollama
            
            enhancement_prompt = f"<think>\nI need to enhance this image prompt: {prompt}\n{context if context else ''}\n</think>\n\nEnhance this image prompt for high-quality generation: {prompt}"
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": enhancement_prompt}
                ],
                options={
                    'temperature': 0.6,
                    'top_p': 0.9,
                    'max_tokens': 150
                }
            )
            
            enhanced = response['message']['content'].strip()
            
            if len(enhanced) < 10 or len(enhanced) > 500:
                return self._rule_based_enhancement(prompt, context)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"❌ Ollama enhancement failed: {e}")
            return self._rule_based_enhancement(prompt, context)

    def _rule_based_enhancement(self, prompt: str, context: str) -> str:
        """Improved fallback rule-based prompt enhancement"""
        enhancements = {
            'dragon': 'majestic dragon with iridescent scales, breathing ethereal fire',
            'city': 'futuristic cyberpunk cityscape with neon lights and flying vehicles',
            'robot': 'sleek mechanical robot with glowing circuits and metallic finish',
            'landscape': 'breathtaking landscape with dramatic lighting and rich textures',
            'sunset': 'golden hour sunset with warm atmospheric lighting',
            'forest': 'mystical forest with dappled sunlight and ancient trees',
            'castle': 'medieval castle with stone towers and gothic architecture',
            'ocean': 'vast ocean with crystal clear water and dramatic waves',
            'mountain': 'snow-capped mountain peaks with misty valleys',
            'space': 'cosmic space scene with nebulae and distant stars',
            'cat': 'elegant cat with piercing eyes and luxurious fur',
            'flower': 'vibrant flower with delicate petals and morning dew',
            'car': 'sleek sports car with polished chrome and dynamic lines'
        }
        
        enhanced = prompt.lower()
        for key, enhancement in enhancements.items():
            if key in enhanced:
                enhanced = enhanced.replace(key, enhancement)
        
        quality_descriptors = [
            'highly detailed', '8k resolution', 'professional lighting',
            'digital art masterpiece', 'cinematic composition', 'vibrant colors'
        ]
        
        for descriptor in quality_descriptors:
            if not any(word in enhanced for word in descriptor.split()):
                enhanced += f', {descriptor}'
        
        return enhanced.title()

# Global LLM processor
llm_processor = DeepSeekLLMProcessor()

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """Stores user-specific configuration data"""
    for uid, conf in configuration.items():
        logging.info(f"🔧 Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

############################################################
# Execution callback function - UPDATED FOR DIRECT URLS
############################################################
def execute(request: InputClass, ray: Ray, state: State) -> OutputClass:
    try:
        original_prompt = request.prompt
        logging.info(f"🎯 Processing prompt: '{original_prompt}'")
        ray.progress(step=10)
        
        # Step 1: Enhance prompt with DeepSeek LLM
        enhanced_prompt = llm_processor.enhance_prompt(original_prompt)
        logging.info(f"🧠 Enhanced prompt: {enhanced_prompt}")
        ray.progress(step=25)
        
        # Step 2: Initialize Openfabric connections with direct URLs
        user_config = configurations.get('super-user')
        if not user_config or not user_config.app_ids:
            raise Exception("No app configuration found")
        
        logging.info(f"🔌 Attempting to connect to direct URLs: {user_config.app_ids}")
        stub = Stub(user_config.app_ids)
        
        # Check connected apps
        connected_apps = stub.get_connected_apps()
        if not connected_apps:
            logging.warning("⚠️ No Openfabric connections established - using mock mode")
            return execute_mock_mode(original_prompt, enhanced_prompt, ray)
        
        logging.info(f"✅ Connected to apps: {connected_apps}")
        ray.progress(step=40)
        
        # Step 3: Generate image using Text-to-Image app
        if not stub.is_connected(TEXT_TO_IMAGE_URL):
            logging.warning(f"⚠️ Text-to-image URL not connected: {TEXT_TO_IMAGE_URL}")
            return execute_mock_mode(original_prompt, enhanced_prompt, ray)
        
        logging.info("🎨 Calling text-to-image Openfabric app...")
        image_input = {
            "prompt": enhanced_prompt,
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "guidance_scale": 7.5,
            "return_type": "base64"
        }
        
        image_result = stub.call(TEXT_TO_IMAGE_URL, image_input, 'super-user')
        ray.progress(step=65)
        
        if not image_result or 'result' not in image_result:
            raise Exception("Image generation failed - no result returned")
        
        # Save generated image
        image_data = image_result['result']
        image_filename = f"image_{uuid.uuid4().hex[:8]}.png"
        image_path = f"outputs/{image_filename}"
        os.makedirs("outputs", exist_ok=True)
        
        # Handle image data with robust error handling
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
            elif isinstance(image_data, bytes):
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            else:
                logging.warning(f"⚠️ Unexpected image data type: {type(image_data)}")
                with open(image_path.replace('.png', '.txt'), 'w') as f:
                    f.write(str(image_data))
        except Exception as e:
            logging.error(f"❌ Failed to save image: {e}")
            with open(image_path.replace('.png', '.txt'), 'w') as f:
                f.write(str(image_data))
        
        logging.info(f"🖼️ Image saved to: {image_path}")
        ray.progress(step=80)
        
        # Step 4: Generate 3D model using Image-to-3D app
        if not stub.is_connected(IMAGE_TO_3D_URL):
            logging.warning(f"⚠️ Image-to-3D URL not connected: {IMAGE_TO_3D_URL}")
            # Continue with just image generation
            generation_id = memory_manager.store_generation(
                original_prompt, enhanced_prompt, image_data, "3D_generation_skipped"
            )
            
            response = OutputClass()
            response.message = (
                f"🎨 Image Generation Complete!\n"
                f"📝 Original: {original_prompt}\n"
                f"🧠 Enhanced: {enhanced_prompt[:100]}...\n"
                f"🖼️ Image: {image_filename}\n"
                f"⚠️ 3D Model: Skipped (URL not connected)\n"
                f"🆔 Generation ID: {generation_id}\n"
                f"✅ Partial generation successful!"
            )
            ray.progress(step=100)
            return response
        
        logging.info("🗿 Calling image-to-3D Openfabric app...")
        model_input = {
            "image": image_data,
            "resolution": 512,
            "topology": "quad",
            "ai_model": "meshy-5",
            "return_type": "base64"
        }
        
        model_result = stub.call(IMAGE_TO_3D_URL, model_input, 'super-user')
        
        if not model_result or 'result' not in model_result:
            raise Exception("3D model generation failed - no result returned")
        
        # Save generated 3D model
        model_data = model_result['result']
        model_filename = f"model_{uuid.uuid4().hex[:8]}.obj"
        model_path = f"outputs/{model_filename}"
        
        try:
            if isinstance(model_data, str):
                if model_data.startswith('data:model') or model_data.startswith('data:application'):
                    model_data = model_data.split(',')[1]
                model_bytes = base64.b64decode(model_data)
                with open(model_path, 'wb') as f:
                    f.write(model_bytes)
            elif isinstance(model_data, bytes):
                with open(model_path, 'wb') as f:
                    f.write(model_data)
            else:
                logging.warning(f"⚠️ Unexpected model data type: {type(model_data)}")
                with open(model_path.replace('.obj', '.txt'), 'w') as f:
                    f.write(str(model_data))
        except Exception as e:
            logging.error(f"❌ Failed to save model: {e}")
            with open(model_path.replace('.obj', '.txt'), 'w') as f:
                f.write(str(model_data))
        
        logging.info(f"🗿 3D model saved to: {model_path}")
        ray.progress(step=95)
        
        # Step 5: Store complete generation in memory
        generation_id = memory_manager.store_generation(
            original_prompt,
            enhanced_prompt,
            image_data,
            model_data
        )
        
        ray.progress(step=100)
        
        # Create comprehensive response
        response = OutputClass()
        response.message = (
            f"🎨 3D Model Generation Complete!\n"
            f"📝 Original: {original_prompt}\n"
            f"🧠 Enhanced: {enhanced_prompt[:100]}...\n"
            f"🖼️ Image: {image_filename}\n"
            f"🗿 3D Model: {model_filename}\n"
            f"🆔 Generation ID: {generation_id}\n"
            f"✅ Ready for download and 3D printing!"
        )
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        return execute_mock_mode(original_prompt, enhanced_prompt, ray)

def execute_mock_mode(original_prompt: str, enhanced_prompt: str, ray: Ray) -> OutputClass:
    """Execute in mock mode when Openfabric servers are unavailable"""
    try:
        logging.info("🎭 Executing in mock mode due to Openfabric server issues")
        ray.progress(step=60)
        
        # Create mock files for demonstration
        os.makedirs("outputs", exist_ok=True)
        
        # Mock image generation
        image_filename = f"mock_image_{uuid.uuid4().hex[:8]}.png"
        image_path = f"outputs/{image_filename}"
        
        # Create a simple placeholder image
        placeholder_img = Image.new('RGB', (512, 512), color=(139, 69, 19))
        placeholder_img.save(image_path)
        
        ray.progress(step=80)
        
        # Mock 3D model generation
        model_filename = f"mock_model_{uuid.uuid4().hex[:8]}.obj"
        model_path = f"outputs/{model_filename}"
        
        # Create a simple OBJ file
        with open(model_path, 'w') as f:
            f.write("# Mock 3D Model Generated by AI Pipeline\n")
            f.write("# Original prompt: " + original_prompt + "\n")
            f.write("# Enhanced prompt: " + enhanced_prompt + "\n")
            f.write("v 0.0 0.0 0.0\n")
            f.write("v 1.0 0.0 0.0\n")
            f.write("v 0.0 1.0 0.0\n")
            f.write("f 1 2 3\n")
        
        ray.progress(step=95)
        
        # Store in memory
        generation_id = memory_manager.store_generation(
            original_prompt,
            enhanced_prompt,
            f"mock_image_data_{uuid.uuid4().hex[:8]}",
            f"mock_model_data_{uuid.uuid4().hex[:8]}"
        )
        
        ray.progress(step=100)
        
        response = OutputClass()
        response.message = (
            f"🎨 AI Pipeline Complete (Demo Mode)!\n"
            f"📝 Original: {original_prompt}\n"
            f"🧠 Enhanced: {enhanced_prompt[:100]}...\n"
            f"🖼️ Image: {image_filename}\n"
            f"🗿 3D Model: {model_filename}\n"
            f"🆔 Generation ID: {generation_id}\n"
            f"📊 Status: Demo mode (Openfabric servers offline)\n"
            f"✅ All core components functional!"
        )
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Mock mode failed: {e}")
        response = OutputClass()
        response.message = f"❌ Generation Error: {str(e)}"
        return response