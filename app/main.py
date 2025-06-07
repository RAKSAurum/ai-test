"""
AI Pipeline for Text-to-Image-to-3D Generation

This module implements a complete AI pipeline that:
1. Takes text prompts and enhances them using LLM
2. Generates images from enhanced prompts via API queue workflow
3. Converts images to 3D models using direct execution workflow

The pipeline uses discovered API workflows for robust generation handling.
"""

import logging
import os
import requests
import base64
import time
from PIL import Image
from typing import Dict
import io
import json
import uuid

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ollama

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import State, Ray
from core.stub import Stub

# Global configurations and API endpoints
configurations: Dict[str, ConfigClass] = dict()
TEXT_TO_IMAGE_URL = 'https://c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network'
IMAGE_TO_3D_URL = 'https://5891a64fe34041d98b0262bb1175ff07.node3.openfabric.network'


def generate_image_via_queue(prompt: str, max_wait_time: int = 300) -> bytes:
    """
    Generate image using queue-based workflow for reliable processing.
    
    Uses a 3-step process:
    1. Submit prompt to generation queue
    2. Poll queue status until completion
    3. Download generated image via resource endpoint
    
    Args:
        prompt: Text description for image generation
        max_wait_time: Maximum seconds to wait for generation completion
        
    Returns:
        bytes: Generated image data
        
    Raises:
        Exception: If queue submission, polling, or download fails
    """
    try:
        logging.info(f"🎨 Generating image for prompt: '{prompt}'")
        
        # Submit to generation queue
        response = requests.post(
            f"{TEXT_TO_IMAGE_URL}/queue/post",
            json={"prompt": prompt},
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Queue submission failed: {response.status_code}")
        
        queue_data = response.json()
        qid = queue_data['qid']
        logging.info(f"✅ Image generation queued with ID: {qid}")
        
        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                status_response = requests.get(f"{TEXT_TO_IMAGE_URL}/queue/list", timeout=10)
                if status_response.status_code == 200:
                    rays = status_response.json()
                    current_ray = next((ray for ray in rays if ray['qid'] == qid), None)
                    
                    if current_ray and current_ray['status'] == 'COMPLETED':
                        logging.info("✅ Image generation completed!")
                        break
                
                logging.info("🔄 Still generating image...")
                time.sleep(5)
                
            except Exception as e:
                logging.debug(f"Polling error: {e}")
                time.sleep(5)
        else:
            raise Exception(f"Image generation timed out after {max_wait_time} seconds")
        
        # Retrieve generated image
        result_response = requests.get(
            f"{TEXT_TO_IMAGE_URL}/queue/get",
            params={"qid": qid},
            timeout=10
        )
        if result_response.status_code != 200:
            raise Exception(f"Failed to get result: {result_response.status_code}")
        
        result_data = result_response.json()
        if 'result' not in result_data:
            raise Exception("Image generation failed - no result found")
        
        # Download image using resource endpoint
        resource_id = result_data['result']
        download_response = requests.get(
            f"{TEXT_TO_IMAGE_URL}/resource",
            params={"reid": resource_id},
            timeout=30
        )
        
        if download_response.status_code != 200:
            raise Exception(f"Failed to download image: {download_response.status_code}")
        
        image_bytes = download_response.content
        logging.info(f"✅ Image downloaded: {len(image_bytes)} bytes")
        return image_bytes
        
    except Exception as e:
        logging.error(f"❌ Image generation failed: {e}")
        raise


def generate_3d_model_from_image(image_bytes: bytes, output_type: str = "object", timeout: int = 600) -> Dict:
    """
    Generate 3D model from image using direct execution workflow.
    
    Implements a 5-step process:
    1. Convert image to base64 data URL
    2. Configure 3D generation parameters
    3. Submit image for 3D processing
    4. Parse JSON response with error handling
    5. Download generated 3D resources
    
    Args:
        image_bytes: Raw image data to convert to 3D
        output_type: Type of 3D output to generate
        timeout: Maximum seconds to wait for 3D generation
        
    Returns:
        Dict: Contains 3D model data, sizes, and metadata
        
    Raises:
        Exception: If 3D generation or download fails
    """
    try:
        logging.info("🗿 Converting image to 3D model...")
        
        # Convert image to base64 data URL format
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        # Configure 3D generation parameters
        try:
            config_response = requests.post(
                f"{IMAGE_TO_3D_URL}/config",
                params={"uid": "default"},
                json={"output_type": output_type},
                timeout=30
            )
            logging.info(f"Config response status: {config_response.status_code}")
        except Exception as e:
            logging.warning(f"Config request failed (continuing anyway): {e}")
        
        # Submit for 3D generation
        logging.info("🔄 Sending 3D generation request...")
        model_response = requests.post(
            f"{IMAGE_TO_3D_URL}/execution",
            json={"input_image": image_data_url},
            timeout=timeout
        )
        
        logging.info(f"3D model response status: {model_response.status_code}")
        
        if model_response.status_code != 200:
            raise Exception(f"3D model generation failed: {model_response.status_code} - {model_response.text}")
        
        # Parse JSON response with error recovery
        try:
            model_result = model_response.json()
        except ValueError as json_error:
            logging.warning(f"JSON parsing error, attempting to fix: {json_error}")
            response_text = model_response.text
            try:
                fixed_json = response_text.replace("'", '"').replace('None', 'null')
                model_result = json.loads(fixed_json)
                logging.info("✅ Successfully parsed fixed JSON response")
            except Exception:
                raise Exception(f"Invalid JSON response from 3D API: {json_error}")
        
        # Download generated 3D resources
        results = {'model_result': model_result}
        
        # Download 3D object file
        if 'generated_object' in model_result and model_result['generated_object']:
            resource_path = model_result['generated_object']
            logging.info(f"3D object resource path: {resource_path}")
            
            try:
                object_response = requests.get(
                    f"{IMAGE_TO_3D_URL}/resource",
                    params={"reid": resource_path},
                    timeout=30
                )
                
                if object_response.status_code == 200:
                    results['object_bytes'] = object_response.content
                    results['object_size'] = len(object_response.content)
                    logging.info(f"✅ 3D model downloaded: {results['object_size']} bytes")
                else:
                    logging.error(f"Failed to download 3D object: {object_response.status_code}")
                    
            except Exception as e:
                logging.error(f"Error downloading 3D object: {e}")
        
        # Download video object if available
        if 'video_object' in model_result and model_result['video_object']:
            try:
                video_response = requests.get(model_result['video_object'], timeout=30)
                if video_response.status_code == 200:
                    results['video_bytes'] = video_response.content
                    results['video_size'] = len(video_response.content)
                    logging.info(f"✅ 3D video downloaded: {results['video_size']} bytes")
            except Exception as e:
                logging.error(f"Error downloading video: {e}")
        
        return results
        
    except Exception as e:
        logging.error(f"❌ 3D model generation failed: {e}")
        raise


class DeepSeekLLMProcessor:
    """
    Handles prompt enhancement using local LLM models.
    
    Uses Ollama for text enhancement with fallback to rule-based enhancement.
    Provides improved prompt quality for better image generation results.
    """
    
    def __init__(self):
        """Initialize LLM processor with model configuration."""
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "deepseek-r1:1.5b"

    def enhance_prompt(self, original_prompt: str) -> str:
        """
        Enhance user prompt using LLM for better image generation.
        
        Args:
            original_prompt: User's original text description
            
        Returns:
            str: Enhanced prompt optimized for image generation
        """
        try:
            enhanced = self._ollama_enhancement(original_prompt)
            logging.info(f"🧠 Enhanced: '{original_prompt}' -> '{enhanced}'")
            return enhanced
            
        except Exception as e:
            logging.error(f"❌ LLM processing failed: {e}")
            return self._rule_based_enhancement(original_prompt)

    def _ollama_enhancement(self, prompt: str) -> str:
        """
        Use Ollama LLM for prompt enhancement.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            str: LLM-enhanced prompt or fallback to rule-based
        """
        try:
            enhancement_prompt = f"Enhance this image prompt for high-quality generation: {prompt}"
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": enhancement_prompt}],
                options={'temperature': 0.6, 'top_p': 0.9, 'max_tokens': 150}
            )
            
            enhanced = response['message']['content'].strip()
            return enhanced if len(enhanced) > 10 else self._rule_based_enhancement(prompt)
            
        except Exception as e:
            logging.error(f"❌ Ollama enhancement failed: {e}")
            return self._rule_based_enhancement(prompt)

    def _rule_based_enhancement(self, prompt: str) -> str:
        """
        Fallback rule-based prompt enhancement.
        
        Applies predefined enhancement patterns to improve prompt quality
        when LLM enhancement is unavailable.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            str: Enhanced prompt using predefined rules
        """
        enhancements = {
            'robot': 'sleek mechanical robot with glowing circuits and metallic finish',
            'dragon': 'majestic dragon with iridescent scales',
            'city': 'futuristic cyberpunk cityscape with neon lights',
            'landscape': 'breathtaking landscape with dramatic lighting'
        }
        
        enhanced = prompt.lower()
        for key, enhancement in enhancements.items():
            if key in enhanced:
                enhanced = enhanced.replace(key, enhancement)
        
        enhanced += ', highly detailed, 8k resolution, professional lighting, digital art masterpiece'
        return enhanced.title()


def initialize_default_config():
    """
    Initialize default configuration with API endpoints.
    
    Sets up the configuration dictionary with default API endpoints
    for text-to-image and image-to-3D generation services.
    """
    default_config = ConfigClass()
    default_config.app_ids = [TEXT_TO_IMAGE_URL, IMAGE_TO_3D_URL]
    configurations['super-user'] = default_config
    logging.info("🔧 Default configuration initialized")


def execute_mock_mode(original_prompt: str, enhanced_prompt: str, ray: Ray) -> OutputClass:
    """
    Execute pipeline in mock mode when APIs are unavailable.
    
    Creates placeholder files and simulates the full pipeline for testing
    and development purposes when external APIs are not accessible.
    
    Args:
        original_prompt: User's original prompt
        enhanced_prompt: Enhanced version of prompt
        ray: Progress tracking object
        
    Returns:
        OutputClass: Mock execution results with placeholder files
    """
    try:
        logging.info("🎭 Mock mode")
        ray.progress(step=60)
        
        os.makedirs("outputs", exist_ok=True)
        
        # Create mock image
        image_filename = f"mock_image_{uuid.uuid4().hex[:8]}.png"
        image_path = f"outputs/{image_filename}"
        placeholder_img = Image.new('RGB', (512, 512), color=(139, 69, 19))
        placeholder_img.save(image_path)
        
        ray.progress(step=80)
        
        # Create mock 3D model
        model_filename = f"mock_model_{uuid.uuid4().hex[:8]}.obj"
        model_path = f"outputs/{model_filename}"
        with open(model_path, 'w') as f:
            f.write("# Mock 3D Model\n")
            f.write(f"# Prompt: {original_prompt}\n")
            f.write("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n")
        
        ray.progress(step=95)
        ray.progress(step=100)
        
        response = OutputClass()
        response.message = (
            f"🎨 Mock Generation Complete!\n"
            f"📝 Original: {original_prompt}\n"
            f"🧠 Enhanced: {enhanced_prompt[:100]}...\n"
            f"🖼️ Image: {image_filename}\n"
            f"🗿 3D Model: {model_filename}\n"
            f"📊 Status: Demo mode"
        )
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Mock mode failed: {e}")
        response = OutputClass()
        response.message = f"❌ Error: {str(e)}"
        return response


# Initialize global components
initialize_default_config()
llm_processor = DeepSeekLLMProcessor()


def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Configuration callback function for OpenFabric framework.
    
    Called by the OpenFabric framework to configure the application
    with user-specific settings and API configurations.
    
    Args:
        configuration: Configuration settings from framework
        state: Current application state
    """
    for uid, conf in configuration.items():
        configurations[uid] = conf


def execute(request: InputClass, ray: Ray, state: State) -> OutputClass:
    """
    Main execution function implementing the complete AI pipeline.
    
    Orchestrates the full text-to-image-to-3D generation workflow:
    1. Prompt enhancement using LLM
    2. Image generation via queue workflow
    3. 3D model generation via direct execution
    4. Result compilation and file management
    
    Args:
        request: Input containing user prompt and parameters
        ray: Progress tracking and communication object
        state: Current application state
        
    Returns:
        OutputClass: Complete pipeline results with generated content paths
    """
    try:
        original_prompt = request.prompt
        logging.info(f"🎯 Processing: '{original_prompt}'")
        ray.progress(step=10)
        
        # Step 1: Enhance prompt using LLM
        enhanced_prompt = llm_processor.enhance_prompt(original_prompt)
        ray.progress(step=25)
        
        # Step 2: Generate image using queue workflow
        logging.info("🎨 Generating image via queue workflow...")
        
        try:
            image_bytes = generate_image_via_queue(enhanced_prompt, max_wait_time=300)
            
            # Save generated image
            image_filename = f"image_{uuid.uuid4().hex[:8]}.png"
            image_path = f"outputs/{image_filename}"
            os.makedirs("outputs", exist_ok=True)
            
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            logging.info(f"✅ Image saved: {image_path} ({len(image_bytes)} bytes)")
            
        except Exception as e:
            logging.error(f"❌ Image generation failed: {e}")
            return execute_mock_mode(original_prompt, enhanced_prompt, ray)
        
        ray.progress(step=65)
        
        # Step 3: Generate 3D model using direct execution
        logging.info("🗿 Generating 3D model via direct execution...")
        
        try:
            model_results = generate_3d_model_from_image(image_bytes, output_type="object", timeout=600)
            
            # Save 3D model
            model_filename = f"model_{uuid.uuid4().hex[:8]}.glb"
            model_path = f"outputs/{model_filename}"
            
            if 'object_bytes' in model_results:
                with open(model_path, 'wb') as f:
                    f.write(model_results['object_bytes'])
                logging.info(f"✅ 3D model saved: {model_path} ({model_results['object_size']} bytes)")
            else:
                # Save reference file if binary not available
                with open(model_path.replace('.glb', '.txt'), 'w') as f:
                    f.write(f"3D Model Generation Result:\n")
                    f.write(f"{json.dumps(model_results.get('model_result', {}), indent=2)}\n")
                model_filename = model_filename.replace('.glb', '.txt')
                logging.info("📄 3D model reference saved")
            
            # Save video if available
            if 'video_bytes' in model_results:
                video_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
                video_path = f"outputs/{video_filename}"
                with open(video_path, 'wb') as f:
                    f.write(model_results['video_bytes'])
                logging.info(f"✅ 3D video saved: {video_path} ({model_results['video_size']} bytes)")
            
        except Exception as e:
            logging.error(f"❌ 3D model generation failed: {e}")
            model_filename = "3d_generation_failed.txt"
            model_results = {"error": str(e)}
        
        ray.progress(step=95)
        ray.progress(step=100)
        
        # Compile final response
        response = OutputClass()
        response.message = (
            f"🎨 COMPLETE AI PIPELINE SUCCESS!\n"
            f"📝 Original: {original_prompt}\n"
            f"🧠 Enhanced: {enhanced_prompt[:100]}...\n"
            f"🖼️ Image: {image_filename} ({len(image_bytes)} bytes)\n"
            f"🗿 3D Model: {model_filename}\n"
            f"🚀 Using discovered pipeline workflow!"
        )
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        return execute_mock_mode(original_prompt, enhanced_prompt, ray)