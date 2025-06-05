import logging
import os
import sqlite3
import json
import base64
from datetime import datetime
from typing import Dict, Optional
import uuid
from PIL import Image

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

# App IDs from the challenge
TEXT_TO_IMAGE_APP_ID = 'f0997a01-d6d3-a5fe-53d8-561300318557'
IMAGE_TO_3D_APP_ID = '69543f29-4d41-4afc-7f29-3d51591f11eb'

def initialize_default_config():
    default_config = ConfigClass()
    default_config.app_ids = [
        f"{TEXT_TO_IMAGE_APP_ID}.node2.openfabric.network",
        f"{IMAGE_TO_3D_APP_ID}.node2.openfabric.network"
    ]
    configurations['super-user'] = default_config
    logging.info("Default configuration initialized")

# Call this at module level
initialize_default_config()

# Memory storage
class MemoryManager:
    def __init__(self):
        self.db_path = "memory/memory.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Don't create connection here - create per thread
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
                    image_data BLOB,
                    model_data BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
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
            ''', (generation_id, original_prompt, enhanced_prompt, image_data, model_data))
            conn.commit()
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
            logging.error(f"Memory recall failed: {e}")
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
        
        # Use local model if provided, otherwise use Ollama
        if local_model_path:
            self.model_name = local_model_path
            self.load_model()
        else:
            self.model_name = "deepseek-r1:1.5b"
            self.model_loaded = False  # Will use Ollama fallback

    def load_model(self):
        """Load DeepSeek model and tokenizer with optimizations"""
        try:
            logging.info(f"Loading DeepSeek model: {self.model_name}")
            logging.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations for better performance
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_4bit=True,  # 4-bit quantization for memory efficiency
                use_cache=True  # Enable KV cache for faster inference
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            logging.info("DeepSeek model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Failed to load DeepSeek model: {e}")
            logging.info("Falling back to Ollama enhancement")
            self.model_loaded = False

    def enhance_prompt(self, original_prompt: str) -> str:
        """
        Enhance the user prompt with DeepSeek LLM or Ollama fallback
        """
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
            
            logging.info(f"Enhanced prompt: '{original_prompt}' -> '{enhanced}'")
            return enhanced
            
        except Exception as e:
            logging.error(f"LLM processing failed: {e}")
            return self._rule_based_enhancement(original_prompt, "")

    def _deepseek_enhancement(self, prompt: str, context: str) -> str:
        """Use DeepSeek model to enhance the prompt with R1 optimizations"""
        try:
            # DeepSeek R1 specific prompt format with thinking trigger
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

            # Tokenize input
            inputs = self.tokenizer(
                enhancement_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate response with R1 optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.6,  # R1 recommended temperature
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract enhanced prompt after "Enhanced prompt:"
            if "Enhanced prompt:" in full_response:
                enhanced_prompt = full_response.split("Enhanced prompt:")[-1].strip()
            else:
                # Fallback extraction
                input_length = len(enhancement_prompt)
                enhanced_prompt = full_response[input_length:].strip()

            # Clean up the response
            enhanced_prompt = enhanced_prompt.replace('\n', ' ').strip()
            enhanced_prompt = enhanced_prompt.split('.')[0] if '.' in enhanced_prompt else enhanced_prompt

            # Ensure we have a reasonable response
            if len(enhanced_prompt) < 10 or len(enhanced_prompt) > 500:
                logging.warning("DeepSeek response seems invalid, using fallback")
                return self._rule_based_enhancement(prompt, context)

            return enhanced_prompt

        except Exception as e:
            logging.error(f"DeepSeek enhancement failed: {e}")
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
            
            # Clean up response
            if len(enhanced) < 10 or len(enhanced) > 500:
                return self._rule_based_enhancement(prompt, context)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Ollama enhancement failed: {e}")
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
        
        # Add artistic qualities with better logic
        quality_descriptors = [
            'highly detailed', '8k resolution', 'professional lighting',
            'digital art masterpiece', 'cinematic composition', 'vibrant colors'
        ]
        
        # Only add descriptors that aren't already present
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
    """
    Stores user-specific configuration data.
    
    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

############################################################
# Execution callback function
############################################################
def execute(request: InputClass, ray: Ray, state: State) -> OutputClass:
    try:
        original_prompt = request.prompt
        logging.info(f"Processing prompt: '{original_prompt}'")
        ray.progress(step=10)
        
        # Step 1: Enhance prompt with DeepSeek LLM
        enhanced_prompt = llm_processor.enhance_prompt(original_prompt)
        logging.info(f"Enhanced prompt: {enhanced_prompt}")
        ray.progress(step=25)
        
        # Step 2: Initialize Openfabric connections
        user_config = configurations.get('super-user')
        if not user_config or not user_config.app_ids:
            raise Exception("No app configuration found")
        
        stub = Stub(user_config.app_ids)
        ray.progress(step=40)
        
        # Step 3: Generate image using Text-to-Image app
        logging.info("Calling text-to-image Openfabric app...")
        image_input = {
            "prompt": enhanced_prompt,
            "width": 512,
            "height": 512,
            "steps": 20,
            "guidance_scale": 7.5
        }
        
        image_result = stub.call(
            f"{TEXT_TO_IMAGE_APP_ID}.node2.openfabric.network",
            image_input,
            'super-user'
        )
        ray.progress(step=65)
        
        if not image_result or 'result' not in image_result:
            raise Exception("Image generation failed - no result returned")
        
        # Save generated image
        image_data = image_result['result']
        image_filename = f"image_{uuid.uuid4().hex[:8]}.png"
        image_path = f"outputs/{image_filename}"
        os.makedirs("outputs", exist_ok=True)
        
        # Handle different image data formats
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
        elif isinstance(image_data, bytes):
            # Raw image bytes
            with open(image_path, 'wb') as f:
                f.write(image_data)
        else:
            # Assume it's a URL or resource reference
            logging.info(f"Image result: {image_data}")
        
        logging.info(f"Image saved to: {image_path}")
        ray.progress(step=80)
        
        # Step 4: Generate 3D model using Image-to-3D app
        logging.info("Calling image-to-3D Openfabric app...")
        model_input = {
            "image": image_data,
            "resolution": 256,
            "steps": 50
        }
        
        model_result = stub.call(
            f"{IMAGE_TO_3D_APP_ID}.node2.openfabric.network",
            model_input,
            'super-user'
        )
        
        if not model_result or 'result' not in model_result:
            raise Exception("3D model generation failed - no result returned")
        
        # Save generated 3D model
        model_data = model_result['result']
        model_filename = f"model_{uuid.uuid4().hex[:8]}.obj"
        model_path = f"outputs/{model_filename}"
        
        if isinstance(model_data, bytes):
            with open(model_path, 'wb') as f:
                f.write(model_data)
        else:
            logging.info(f"3D model result: {model_data}")
        
        logging.info(f"3D model saved to: {model_path}")
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
        logging.error(f"Pipeline failed: {e}")
        response = OutputClass()
        response.message = f"❌ Generation Error: {str(e)}"
        return response