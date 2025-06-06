# Create test_api.py
import requests
import json

def test_openfabric_input():
    url = "https://c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network/execution"
    
    # Test different input combinations
    test_cases = [
        {"prompt": "test robot"},
        {"prompt": "test robot", "width": 512},
        {"prompt": "test robot", "width": 512, "height": 512},
        {"prompt": "test robot", "width": 1024, "height": 1024},
        {"prompt": "test robot", "width": 1024, "height": 1024, "steps": 20},
        {"prompt": "test robot", "width": 1024, "height": 1024, "steps": 30, "guidance_scale": 7.5},
        # Your current full input
        {
            "prompt": "test robot",
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "guidance_scale": 7.5,
            "return_type": "base64"
        }
    ]
    
    for i, test_input in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}: {list(test_input.keys())}")
        try:
            response = requests.post(url, json=test_input, timeout=30)
            print(f"✅ Status: {response.status_code}")
            if response.status_code == 200:
                print(f"🎉 SUCCESS with fields: {list(test_input.keys())}")
                break
            elif response.status_code == 422:
                print(f"❌ 422 Error with fields: {list(test_input.keys())}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_openfabric_input()
