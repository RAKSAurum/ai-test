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
        """Make API call to the remote app - FIXED JSON HANDLING"""
        if not self.connected:
            raise Exception(f"Not connected to {self.url}")
        
        try:
            # FIXED: Use /execution endpoint with clean input data
            logging.info(f"🔄 Calling {self.url}/execution with input keys: {list(input_data.keys())}")
            
            response = requests.post(
                f"{self.url}/execution",  # Correct endpoint
                json=input_data,  # Use original input_data without extra fields
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logging.info(f"✅ API call successful: {self.url}/execution")
                
                # FIXED: Handle JSON parsing errors gracefully
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    logging.error(f"❌ Invalid JSON response: {response.text[:200]}")
                    # Return response text as result for non-JSON responses
                    return {"result": response.text}
            else:
                logging.error(f"❌ API call failed: {response.status_code} - {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")
                
        except Exception as e:
            logging.error(f"❌ Remote call failed for {self.url}: {e}")
            raise

class Stub:
    def __init__(self, app_urls: List[str]):
        self._schema: Dict[str, tuple] = {}
        self._manifest: Dict[str, dict] = {}
        self._connections: Dict[str, Remote] = {}
        self.apps = app_urls

        logging.info(f"🔍 Testing connectivity for direct URLs: {app_urls}")
        
        for app_url in app_urls:
            # Clean and normalize the URL
            app_url = app_url.strip('/')
            if not app_url.startswith('http'):
                app_url = f"http://{app_url}"
            
            # Store normalized URL for consistent lookup
            normalized_url = app_url.rstrip('/')
            connected = False
            
            try:
                logging.info(f"🧪 Testing direct URL: {app_url}")
                
                # Test manifest endpoint
                manifest_response = requests.get(f"{app_url}/manifest", timeout=10)
                status_code = manifest_response.status_code
                
                logging.info(f"[{app_url}] Manifest response: {status_code}")
                
                if status_code == 200:
                    try:
                        manifest = manifest_response.json()
                        self._manifest[normalized_url] = manifest
                        logging.info(f"📋 Manifest loaded for {app_url}")
                        
                        # Try to get schemas if available
                        try:
                            input_response = requests.get(f"{app_url}/schema?type=input", timeout=5)
                            output_response = requests.get(f"{app_url}/schema?type=output", timeout=5)
                            
                            if input_response.status_code == 200 and output_response.status_code == 200:
                                input_schema = input_response.json()
                                output_schema = output_response.json()
                                self._schema[normalized_url] = (input_schema, output_schema)
                                logging.info(f"📋 Schemas loaded for {app_url}")
                        except:
                            logging.debug(f"⚠️ Schemas not available for {app_url}")
                        
                        # Establish connection using normalized URL as key
                        self._connections[normalized_url] = Remote(app_url, f"{app_url}-proxy").connect()
                        
                        if self._connections[normalized_url].connected:
                            logging.info(f"🎯 Successfully connected to {app_url}")
                            connected = True
                        
                    except json.JSONDecodeError:
                        logging.warning(f"⚠️ Invalid JSON response from {app_url}")
                        continue
                        
                elif status_code in [502, 503, 504]:
                    logging.warning(f"🔧 Server error {status_code} for {app_url} - may be temporarily down")
                elif status_code == 404:
                    logging.debug(f"🔍 Not found: {app_url}")
                else:
                    logging.debug(f"❓ Unexpected status {status_code} for {app_url}")
                    
            except requests.exceptions.ConnectionError as e:
                if "Name or service not known" in str(e):
                    logging.debug(f"🌐 DNS resolution failed for {app_url}")
                else:
                    logging.debug(f"🔌 Connection error for {app_url}: {e}")
            except requests.exceptions.Timeout:
                logging.debug(f"⏰ Timeout for {app_url}")
            except Exception as e:
                logging.error(f"❌ Unexpected error with {app_url}: {e}")
            
            if not connected:
                logging.error(f"❌ Failed to connect to {app_url}")

        # Summary
        connected_apps = self.get_connected_apps()
        if connected_apps:
            logging.info(f"✅ Successfully connected to {len(connected_apps)} apps: {connected_apps}")
        else:
            logging.warning("⚠️ No apps connected - will use fallback mode")

    def call(self, app_url: str, input_data: dict, user_id: str) -> dict:
        """Call an Openfabric app with input data - FIXED: CLEAN INPUT"""
        # Normalize URL for consistent lookup
        normalized_url = app_url.rstrip('/')
        
        if normalized_url not in self._connections:
            raise Exception(f"Connection not found for app URL: {normalized_url}")
        
        connection = self._connections[normalized_url]
        if not connection.connected:
            raise Exception(f"Connection not established for app URL: {normalized_url}")
        
        try:
            # FIXED: Pass original input_data without extra fields that cause 422 error
            response = connection.call(input_data)  # Removed enhanced_input
            logging.info(f"✅ [{normalized_url}] Call successful")
            return response
        except Exception as e:
            logging.error(f"❌ [{normalized_url}] Call failed: {e}")
            raise

    def get_connected_apps(self) -> List[str]:
        """Get list of successfully connected apps"""
        return [app_url for app_url, conn in self._connections.items() if conn.connected]

    def is_connected(self, app_url: str) -> bool:
        """Check if specific app is connected"""
        # Normalize URL for consistent lookup
        normalized_url = app_url.rstrip('/')
        return normalized_url in self._connections and self._connections[normalized_url].connected

    def get_manifest(self, app_url: str) -> Optional[dict]:
        """Get manifest for specific app"""
        normalized_url = app_url.rstrip('/')
        return self._manifest.get(normalized_url)

    def get_schema(self, app_url: str) -> Optional[tuple]:
        """Get input/output schemas for specific app"""
        normalized_url = app_url.rstrip('/')
        return self._schema.get(normalized_url)