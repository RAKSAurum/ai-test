import requests
import logging
from typing import List, Dict, Optional
import json
import time

class Remote:
    def __init__(self, url: str, proxy_name: str):
        self.url = url
        self.proxy_name = proxy_name
        self.connected = False

    def connect(self):
        """Establish connection to remote app"""
        try:
            # Test connection with manifest endpoint
            response = requests.get(f"{self.url}/manifest", timeout=10)
            self.connected = response.status_code == 200
            if self.connected:
                logging.info(f"✅ Connected to {self.url}")
            else:
                logging.warning(f"⚠️ Connection failed: {self.url} returned {response.status_code}")
            return self
        except Exception as e:
            logging.error(f"❌ Failed to connect to {self.url}: {e}")
            self.connected = False
            return self

    def call(self, input_data: dict):
        """Make API call to the remote app"""
        if not self.connected:
            raise Exception(f"Not connected to {self.url}")
        
        try:
            logging.info(f"🔄 Calling {self.url}/execute with input keys: {list(input_data.keys())}")
            
            response = requests.post(
                f"{self.url}/execute",
                json=input_data,
                timeout=60,  # Increased timeout for AI processing
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logging.info(f"✅ API call successful: {self.url}")
                return response.json()
            else:
                logging.error(f"❌ API call failed: {response.status_code} - {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")
                
        except Exception as e:
            logging.error(f"❌ Remote call failed for {self.url}: {e}")
            raise

class Stub:
    def __init__(self, app_ids: List[str]):
        self._schema: Dict[str, tuple] = {}
        self._manifest: Dict[str, dict] = {}
        self._connections: Dict[str, Remote] = {}
        self.apps = app_ids

        # Comprehensive URL patterns based on Openfabric documentation[2]
        url_patterns = [
            "https://{}.openfabric.ai",
            "https://api.openfabric.ai/app/{}",
            "https://apps.openfabric.ai/{}",
            "https://{}.node1.openfabric.network",
            "https://{}.node2.openfabric.network", 
            "https://openfabric.ai/app/{}",
            "https://openfabric.network/app/{}",
            "https://marketplace.openfabric.ai/app/{}",
            "https://{}.openfabric.network"
        ]

        logging.info(f"🔍 Testing connectivity for apps: {app_ids}")
        
        for app_id in app_ids:
            app_id = app_id.strip('/')
            connected = False
            
            for pattern in url_patterns:
                base_url = pattern.format(app_id)
                try:
                    logging.debug(f"🧪 Testing {base_url}")
                    
                    # Test manifest endpoint first
                    manifest_response = requests.get(f"{base_url}/manifest", timeout=10)
                    status_code = manifest_response.status_code
                    
                    logging.info(f"[{app_id}] {base_url}: {status_code}")
                    
                    if status_code == 200:
                        try:
                            manifest = manifest_response.json()
                            self._manifest[app_id] = manifest
                            
                            # Try to get schemas if available
                            try:
                                input_response = requests.get(f"{base_url}/schema?type=input", timeout=5)
                                output_response = requests.get(f"{base_url}/schema?type=output", timeout=5)
                                
                                if input_response.status_code == 200 and output_response.status_code == 200:
                                    input_schema = input_response.json()
                                    output_schema = output_response.json()
                                    self._schema[app_id] = (input_schema, output_schema)
                                    logging.info(f"📋 Schemas loaded for {app_id}")
                            except:
                                logging.debug(f"⚠️ Schemas not available for {app_id}")
                            
                            # Establish connection
                            self._connections[app_id] = Remote(base_url, f"{app_id}-proxy").connect()
                            
                            if self._connections[app_id].connected:
                                logging.info(f"🎯 Successfully connected to {app_id} via {base_url}")
                                connected = True
                                break
                            
                        except json.JSONDecodeError:
                            logging.warning(f"⚠️ Invalid JSON response from {base_url}")
                            continue
                    elif status_code in [502, 503, 504]:
                        logging.warning(f"🔧 Server error {status_code} for {base_url} - may be temporarily down")
                    elif status_code == 404:
                        logging.debug(f"🔍 Not found: {base_url}")
                    else:
                        logging.debug(f"❓ Unexpected status {status_code} for {base_url}")
                        
                except requests.exceptions.ConnectionError as e:
                    if "Name or service not known" in str(e):
                        logging.debug(f"🌐 DNS resolution failed for {base_url}")
                    else:
                        logging.debug(f"🔌 Connection error for {base_url}: {e}")
                except requests.exceptions.Timeout:
                    logging.debug(f"⏰ Timeout for {base_url}")
                except Exception as e:
                    logging.error(f"❌ Unexpected error with {base_url}: {e}")
            
            if not connected:
                logging.error(f"❌ [{app_id}] Failed to connect via any URL pattern")

        # Summary
        connected_apps = self.get_connected_apps()
        if connected_apps:
            logging.info(f"✅ Successfully connected to {len(connected_apps)} apps: {connected_apps}")
        else:
            logging.warning("⚠️ No apps connected - will use fallback mode")

    def call(self, app_id: str, input_data: dict, user_id: str) -> dict:
        """Call an Openfabric app with input data"""
        if app_id not in self._connections:
            raise Exception(f"Connection not found for app ID: {app_id}")
        
        connection = self._connections[app_id]
        if not connection.connected:
            raise Exception(f"Connection not established for app ID: {app_id}")
        
        # Enhanced input with user context
        enhanced_input = {
            **input_data,
            "user_id": user_id,
            "app_id": app_id,
            "timestamp": time.time()
        }
        
        try:
            response = connection.call(enhanced_input)
            logging.info(f"✅ [{app_id}] Call successful")
            return response
        except Exception as e:
            logging.error(f"❌ [{app_id}] Call failed: {e}")
            raise

    def get_connected_apps(self) -> List[str]:
        """Get list of successfully connected apps"""
        return [app_id for app_id, conn in self._connections.items() if conn.connected]

    def is_connected(self, app_id: str) -> bool:
        """Check if specific app is connected"""
        return app_id in self._connections and self._connections[app_id].connected

    def get_manifest(self, app_id: str) -> Optional[dict]:
        """Get manifest for specific app"""
        return self._manifest.get(app_id)

    def get_schema(self, app_id: str) -> Optional[tuple]:
        """Get input/output schemas for specific app"""
        return self._schema.get(app_id)