import requests
import time
import base64
import json
import io
from PIL import Image

# Configuration
image_gen_base_url = "https://c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"
model_3d_base_url = "https://5891a64fe34041d98b0262bb1175ff07.node3.openfabric.network"

def generate_image_and_convert_to_3d(prompt, output_type="video"):
    """
    Complete pipeline: Generate image from prompt, then convert to 3D model
    
    Args:
        prompt (str): Text prompt for image generation
        output_type (str): "video", "object", or "both" for 3D output type
    
    Returns:
        dict: Contains paths to generated files
    """
    
    # Step 1: Generate image using your existing code
    print(f"Generating image for prompt: '{prompt}'")
    
    # Submit image generation request to queue
    response = requests.post(f"{image_gen_base_url}/queue/post", json={
        "prompt": prompt
    })
    queue_data = response.json()
    qid = queue_data['qid']
    print(f"Image generation queued with ID: {qid}")
    
    # Poll for image generation completion
    while True:
        status_response = requests.get(f"{image_gen_base_url}/queue/list")
        rays = status_response.json()
        
        current_ray = next((ray for ray in rays if ray['qid'] == qid), None)
        if current_ray and current_ray['status'] == 'COMPLETED':
            break
        
        print("Still generating image...")
        time.sleep(5)
    
    # Get the generated image
    result_response = requests.get(f"{image_gen_base_url}/queue/get", params={"qid": qid})
    result_data = result_response.json()
    
    if 'result' not in result_data:
        raise Exception("Image generation failed - no result found")
    
    # Download the generated image
    resource_id = result_data['result']
    download_response = requests.get(f"{image_gen_base_url}/resource", params={"reid": resource_id})
    
    if download_response.status_code != 200:
        raise Exception(f"Failed to download image: {download_response.status_code}")
    
    # Save the generated image temporarily
    image_filename = f"generated_image_{int(time.time())}.png"
    with open(image_filename, "wb") as f:
        f.write(download_response.content)
    
    print(f"Image saved as: {image_filename}")
    
    # Step 2: Convert image to base64 for 3D model generation
    with open(image_filename, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        # Add data URL prefix for the API
        image_data_url = f"data:image/png;base64,{image_base64}"
    
    print("Converting image to 3D model...")
    
    # Step 3: Configure 3D model generation (optional)
    try:
        config_response = requests.post(f"{model_3d_base_url}/config", 
                                      params={"uid": "default"},
                                      json={"output_type": output_type})
        print(f"Config response status: {config_response.status_code}")
        if config_response.status_code != 200:
            print(f"Config response text: {config_response.text}")
    except Exception as e:
        print(f"Config request failed: {e}")
    
    # Step 4: Generate 3D model
    print("Sending 3D generation request...")
    model_response = requests.post(f"{model_3d_base_url}/execution", json={
        "input_image": image_data_url
    })
    
    print(f"3D model response status: {model_response.status_code}")
    print(f"3D model response headers: {dict(model_response.headers)}")
    print(f"3D model response text: {model_response.text[:500]}...")  # First 500 chars
    
    if model_response.status_code != 200:
        raise Exception(f"3D model generation failed: {model_response.status_code} - {model_response.text}")
    
    # Try to parse JSON with better error handling
    try:
        model_result = model_response.json()
    except ValueError as json_error:
        print(f"JSON parsing error: {json_error}")
        print("Attempting to fix malformed JSON response...")
        
        # The API returns Python dict format instead of JSON
        # Convert single quotes to double quotes for proper JSON
        response_text = model_response.text
        try:
            # Replace single quotes with double quotes, handle None values
            fixed_json = response_text.replace("'", '"').replace('None', 'null')
            model_result = json.loads(fixed_json)
            print("Successfully parsed fixed JSON response")
        except Exception as fix_error:
            print(f"Failed to fix JSON: {fix_error}")
            print(f"Raw response content: {model_response.content}")
            raise Exception(f"Invalid JSON response from 3D API: {json_error}")
    
    # Step 5: Download the 3D model results (FIXED - removed duplication)
    results = {}
    
    # Download 3D object if available
    if 'generated_object' in model_result and model_result['generated_object']:
        resource_path = model_result['generated_object']
        print(f"3D object resource path: {resource_path}")
        
        try:
            object_response = requests.get(f"{model_3d_base_url}/resource", 
                                         params={"reid": resource_path})
            
            if object_response.status_code == 200:
                object_filename = f"3d_model_{int(time.time())}.glb"
                with open(object_filename, "wb") as f:
                    f.write(object_response.content)
                results['object'] = object_filename
                print(f"3D model object saved as: {object_filename}")
            else:
                print(f"Failed to download 3D object: {object_response.status_code}")
                print(f"Response: {object_response.text}")
        except Exception as e:
            print(f"Error downloading 3D object: {e}")
    
    # Download video if available
    if 'video_object' in model_result and model_result['video_object']:
        try:
            video_response = requests.get(model_result['video_object'])
            if video_response.status_code == 200:
                video_filename = f"3d_model_video_{int(time.time())}.mp4"
                with open(video_filename, "wb") as f:
                    f.write(video_response.content)
                results['video'] = video_filename
                print(f"3D model video saved as: {video_filename}")
        except Exception as e:
            print(f"Error downloading video: {e}")
    else:
        print("No video object in response (video_object is None)")
    
    results['original_image'] = image_filename
    return results

def generate_3d_from_existing_image(image_path, output_type="video"):
    """
    Generate 3D model from an existing image file
    
    Args:
        image_path (str): Path to existing image file
        output_type (str): "video", "object", or "both"
    
    Returns:
        dict: Contains paths to generated files
    """
    
    # Convert existing image to base64
    with open(image_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Detect image format
        image_format = "png"
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            image_format = "jpeg"
            
        image_data_url = f"data:image/{image_format};base64,{image_base64}"
    
    print(f"Converting existing image '{image_path}' to 3D model...")
    
    # Configure 3D model generation
    try:
        config_response = requests.post(f"{model_3d_base_url}/config", 
                                      params={"uid": "default"},
                                      json={"output_type": output_type})
        print(f"Config response status: {config_response.status_code}")
        if config_response.status_code != 200:
            print(f"Config response text: {config_response.text}")
    except Exception as e:
        print(f"Config request failed: {e}")
    
    # Generate 3D model
    print("Sending 3D generation request...")
    model_response = requests.post(f"{model_3d_base_url}/execution", json={
        "input_image": image_data_url
    })
    
    print(f"3D model response status: {model_response.status_code}")
    print(f"3D model response headers: {dict(model_response.headers)}")
    print(f"3D model response text: {model_response.text[:500]}...")  # First 500 chars
    
    if model_response.status_code != 200:
        raise Exception(f"3D model generation failed: {model_response.status_code} - {model_response.text}")
    
    # Try to parse JSON with better error handling
    try:
        model_result = model_response.json()
    except ValueError as json_error:
        print(f"JSON parsing error: {json_error}")
        print(f"Raw response content: {model_response.content}")
        raise Exception(f"Invalid JSON response from 3D API: {json_error}")
    
    # Download results
    results = {}
    
    if 'video_object' in model_result and model_result['video_object']:
        video_response = requests.get(model_result['video_object'])
        video_filename = f"3d_model_video_{int(time.time())}.mp4"
        with open(video_filename, "wb") as f:
            f.write(video_response.content)
        results['video'] = video_filename
        print(f"3D model video saved as: {video_filename}")
    
    if 'generated_object' in model_result and model_result['generated_object']:
        object_response = requests.get(model_result['generated_object'])
        object_filename = f"3d_model_{int(time.time())}.glb"
        with open(object_filename, "wb") as f:
            f.write(object_response.content)
        results['object'] = object_filename
        print(f"3D model object saved as: {object_filename}")
    
    return results

# Example usage
if __name__ == "__main__":
    try:
        # Method 1: Generate image from prompt and convert to 3D
        results = generate_image_and_convert_to_3d(
            prompt="A futuristic robot with glowing blue eyes and metallic armor",
            output_type="both"  # Get both video and 3D object file
        )
        
        print("\n=== Generation Complete ===")
        print(f"Original image: {results['original_image']}")
        if 'video' in results:
            print(f"3D model video: {results['video']}")
        if 'object' in results:
            print(f"3D model file: {results['object']}")
        
        # Method 2: Convert existing image to 3D (uncomment to use)
        # results = generate_3d_from_existing_image("your_image.png", output_type="both")
        
    except Exception as e:
        print(f"Error: {e}")

# Debug function to test 3D API directly
def debug_3d_api():
    """Simple debug function to test the 3D API directly"""
    
    # Test 1: Check if API is accessible
    try:
        manifest_response = requests.get(f"{model_3d_base_url}/manifest")
        print(f"Manifest response status: {manifest_response.status_code}")
        print(f"Manifest response: {manifest_response.text}")
    except Exception as e:
        print(f"Manifest request failed: {e}")
    
    # Test 2: Try a simple image (you'll need to replace with actual base64)
    test_image_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="  # 1x1 pixel
    
    try:
        response = requests.post(f"{model_3d_base_url}/execution", 
                               json={"input_image": test_image_b64},
                               headers={'Content-Type': 'application/json'})
        
        print(f"Test execution status: {response.status_code}")
        print(f"Test execution headers: {dict(response.headers)}")
        print(f"Test execution response: {response.text}")
    except Exception as e:
        print(f"Test execution failed: {e}")

# Uncomment this line to run debug function
# debug_3d_api()