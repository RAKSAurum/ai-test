import logging
import os
import sqlite3
import json
import base64
from datetime import datetime
from typing import Dict, Optional
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import State
from core.stub import Stub

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

# App IDs from the challenge
TEXT_TO_IMAGE_APP_ID = 'f0997a01-d6d3-a5fe-53d8-561300318557'
IMAGE_TO_3D_APP_ID = '69543f29-4d41-4afc-7f29-3d51591f11eb'

# Memory storage
class MemoryManager:
    def __init__(self):
        self.session_memory = {}  # Short-term memory
        self.setup_long_term_memory()
    
    def setup_long_term_memory(self):
        """Initialize SQLite database for long-term memory"""
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('memory', exist_ok=True)
        
        self.conn = sqlite3.connect('memory/generations.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                original_prompt TEXT,
                enhanced_prompt TEXT,
                image_path TEXT,
                model_path TEXT,
                metadata TEXT
            )
        ''')
        self.conn.commit()
    
    def store_generation(self, original_prompt: str, enhanced_prompt: str, 
                        image_data: bytes, model_data: bytes) -> str:
        """Store a complete generation cycle"""
        generation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Save files
        image_path = f"outputs/image_{generation_id}.png"
        model_path = f"outputs/model_{generation_id}.obj"
        
        try:
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            # Store in database
            metadata = json.dumps({
                'generation_id': generation_id,
                'pipeline_version': '1.0',
                'success': True
            })
            
            self.conn.execute('''
                INSERT INTO generations (id, timestamp, original_prompt, enhanced_prompt, 
                                       image_path, model_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (generation_id, timestamp, original_prompt, enhanced_prompt, 
                  image_path, model_path, metadata))
            self.conn.commit()
            
            # Store in session memory
            self.session_memory[generation_id] = {
                'timestamp': timestamp,
                'original_prompt': original_prompt,
                'enhanced_prompt': enhanced_prompt,
                'image_path': image_path,
                'model_path': model_path
            }
            
            return generation_id
            
        except Exception as e:
            logging.error(f"Failed to store generation: {e}")
            return None
    
    def recall_similar(self, prompt: str, limit: int = 5) -> list:
        """Recall similar generations from memory"""
        cursor = self.conn.execute('''
            SELECT id, timestamp, original_prompt, enhanced_prompt, image_path, model_path
            FROM generations 
            WHERE original_prompt LIKE ? OR enhanced_prompt LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (f'%{prompt}%', f'%{prompt}%', limit))
        
        return cursor.fetchall()

# Global memory manager
memory_manager = MemoryManager()

class DeepSeekLLMProcessor:
    """Handles DeepSeek local LLM processing for prompt enhancement"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configuration
        self.model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        # Alternative larger model: "deepseek-ai/deepseek-coder-6.7b-instruct"
        
        self.load_model()
    
    def load_model(self):
        """Load DeepSeek model and tokenizer"""
        try:
            logging.info(f"Loading DeepSeek model: {self.model_name}")
            logging.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
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
            logging.info("Falling back to rule-based enhancement")
            self.model_loaded = False
    
    def enhance_prompt(self, original_prompt: str) -> str:
        """
        Enhance the user prompt with DeepSeek LLM
        """
        try:
            # Check memory for similar prompts first
            similar = memory_manager.recall_similar(original_prompt, 3)
            context = ""
            if similar:
                context = f"Previous similar generations: {[s[2] for s in similar[:2]]}"
            
            if self.model_loaded:
                enhanced = self._deepseek_enhancement(original_prompt, context)
            else:
                enhanced = self._rule_based_enhancement(original_prompt, context)
            
            logging.info(f"Enhanced prompt: '{original_prompt}' -> '{enhanced}'")
            return enhanced
            
        except Exception as e:
            logging.error(f"LLM processing failed: {e}")
            return self._rule_based_enhancement(original_prompt, "")
    
    def _deepseek_enhancement(self, prompt: str, context: str) -> str:
        """Use DeepSeek model to enhance the prompt"""
        try:
            # Create system message for prompt enhancement
            system_prompt = """You are a creative AI assistant specialized in enhancing text-to-image prompts. Your task is to take simple prompts and expand them into detailed, artistic descriptions suitable for high-quality image generation.

Focus on:
- Visual details (colors, textures, lighting)
- Artistic style and quality descriptors
- Atmospheric elements
- Technical photography/art terms
- Keep the core concept but make it vivid and detailed

Example:
Input: "dragon on cliff"
Output: "Majestic dragon with iridescent emerald scales perched on a rocky cliff overlooking a vast valley, golden hour lighting, dramatic clouds, highly detailed digital art, 8k resolution, fantasy masterpiece"

Respond with only the enhanced prompt, no explanations."""
            
            # Format the conversation for DeepSeek
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this image prompt: {prompt}"}
            ]
            
            # Add context if available
            if context:
                messages.append({"role": "user", "content": f"Context: {context}"})
            
            # Apply chat template
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "assistant" in full_response:
                enhanced_prompt = full_response.split("assistant")[-1].strip()
            else:
                # Fallback extraction
                input_length = len(formatted_input)
                enhanced_prompt = full_response[input_length:].strip()
            
            # Clean up the response
            enhanced_prompt = enhanced_prompt.replace('\n', ' ').strip()
            
            # Ensure we have a reasonable response
            if len(enhanced_prompt) < 10 or len(enhanced_prompt) > 500:
                logging.warning("DeepSeek response seems invalid, using fallback")
                return self._rule_based_enhancement(prompt, context)
            
            return enhanced_prompt
            
        except Exception as e:
            logging.error(f"DeepSeek enhancement failed: {e}")
            return self._rule_based_enhancement(prompt, context)
    
    def _rule_based_enhancement(self, prompt: str, context: str) -> str:
        """Fallback rule-based prompt enhancement"""
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
            'space': 'cosmic space scene with nebulae and distant stars'
        }
        
        enhanced = prompt.lower()
        for key, enhancement in enhancements.items():
            if key in enhanced:
                enhanced = enhanced.replace(key, enhancement)
        
        # Add artistic qualities
        quality_descriptors = [
            'highly detailed', '8k resolution', 'professional lighting',
            'digital art masterpiece', 'cinematic composition', 'vibrant colors'
        ]
        
        for descriptor in quality_descriptors:
            if descriptor.split()[0] not in enhanced:
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
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """

    # Retrieve input
    request: InputClass = model.request
    original_prompt = request.prompt
    
    # Retrieve user config
    user_config: ConfigClass = configurations.get('super-user', None)
    logging.info(f"User config: {configurations}")

    # Initialize the Stub with app IDs
    app_ids = user_config.app_ids if user_config else []
    stub = Stub(app_ids)

    # ------------------------------
    # AI PIPELINE IMPLEMENTATION WITH DEEPSEEK
    # ------------------------------
    
    try:
        logging.info(f"Starting AI pipeline for prompt: '{original_prompt}'")
        
        # Step 1: Enhance prompt with DeepSeek LLM
        logging.info("Step 1: Processing with DeepSeek LLM...")
        enhanced_prompt = llm_processor.enhance_prompt(original_prompt)
        
        # Step 2: Generate image using Text-to-Image
        logging.info("Step 2: Generating image...")
        image_result = stub.call(
            TEXT_TO_IMAGE_APP_ID, 
            {'prompt': enhanced_prompt}, 
            'super-user'
        )
        
        if not image_result or 'result' not in image_result:
            raise Exception("Text-to-Image generation failed")
        
        image_data = image_result.get('result')
        logging.info(f"Image generated successfully, size: {len(image_data)} bytes")
        
        # Step 3: Convert image to 3D model
        logging.info("Step 3: Converting image to 3D model...")
        model_3d_result = stub.call(
            IMAGE_TO_3D_APP_ID,
            {'image': image_data},
            'super-user'
        )
        
        if not model_3d_result or 'result' not in model_3d_result:
            raise Exception("Image-to-3D conversion failed")
        
        model_3d_data = model_3d_result.get('result')
        logging.info(f"3D model generated successfully, size: {len(model_3d_data)} bytes")
        
        # Step 4: Store in memory
        logging.info("Step 4: Storing generation in memory...")
        generation_id = memory_manager.store_generation(
            original_prompt, 
            enhanced_prompt, 
            image_data, 
            model_3d_data
        )
        
        if generation_id:
            success_message = (
                f"🎉 Successfully created 3D model from '{original_prompt}'!\n"
                f"🧠 DeepSeek enhanced: {enhanced_prompt}\n"
                f"🆔 Generation ID: {generation_id}\n"
                f"💾 Files saved to outputs/ directory\n"
                f"🔥 Model: {llm_processor.model_name}\n"
                f"⚡ Device: {llm_processor.device}\n"
                f"💭 Stored in memory for future reference"
            )
            logging.info(f"Pipeline completed successfully: {generation_id}")
        else:
            success_message = (
                f"⚠️ 3D model created but storage failed for '{original_prompt}'\n"
                f"🧠 DeepSeek enhanced: {enhanced_prompt}\n"
                f"Pipeline completed with warnings"
            )
            
    except Exception as e:
        error_message = f"❌ Pipeline failed for '{original_prompt}': {str(e)}"
        logging.error(error_message)
        success_message = error_message

    # Prepare response
    response: OutputClass = model.response
    response.message = success_message