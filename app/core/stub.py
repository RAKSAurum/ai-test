import requests
import logging
from typing import List, Dict, Optional
import json

class Remote:
    def __init__(self, url: str, proxy_name: str):
        self.url = url
        self.proxy_name = proxy_name
        self.connected = False

    def connect(self):
        """Establish connection to remote app"""
        try:
            response = requests.get(f"{self.url}/manifest", timeout=10)
            self.connected = response.status_code == 200
            if self.connected:
                logging.info(f"✅ Connected to {self.url}")
            return self
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            self.connected = False
            return self

    def call(self, input_data: dict):
        """Make API call to the remote app - SIMPLIFIED FOR COMPATIBILITY"""
        if not self.connected:
            raise Exception(f"Not connected to {self.url}")
        
        try:
            response = requests.post(
                f"{self.url}/execution",
                json=input_data,
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logging.info(f"✅ API call successful")
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"result": response.text}
            else:
                raise Exception(f"API call failed: {response.status_code}")
                
        except Exception as e:
            logging.error(f"❌ API call failed: {e}")
            raise

class Stub:
    def __init__(self, app_urls: List[str]):
        self._connections: Dict[str, Remote] = {}
        self.apps = app_urls

        for app_url in app_urls:
            app_url = app_url.strip('/').replace('http://', 'https://')
            if not app_url.startswith('https://'):
                app_url = f"https://{app_url}"
            
            normalized_url = app_url.rstrip('/')
            
            try:
                manifest_response = requests.get(f"{app_url}/manifest", timeout=10)
                if manifest_response.status_code == 200:
                    self._connections[normalized_url] = Remote(app_url, f"{app_url}-proxy").connect()
                    if self._connections[normalized_url].connected:
                        logging.info(f"🎯 Connected to {app_url}")
            except Exception as e:
                logging.error(f"❌ Failed to connect to {app_url}: {e}")

        connected_apps = self.get_connected_apps()
        if connected_apps:
            logging.info(f"✅ Connected to {len(connected_apps)} apps")
        else:
            logging.warning("⚠️ No apps connected - using direct API calls")

    def call(self, app_url: str, input_data: dict, user_id: str) -> dict:
        """Call an Openfabric app with input data - COMPATIBILITY WRAPPER"""
        normalized_url = app_url.rstrip('/')
        
        # For compatibility, but main.py now uses direct API calls
        if normalized_url not in self._connections:
            logging.warning(f"Connection not found for: {normalized_url}, using direct calls")
            return {"result": "using_direct_api_calls"}
        
        connection = self._connections[normalized_url]
        if not connection.connected:
            logging.warning(f"Not connected to: {normalized_url}, using direct calls")
            return {"result": "using_direct_api_calls"}
        
        try:
            response = connection.call(input_data)
            return response
        except Exception as e:
            logging.error(f"❌ Call failed: {e}")
            raise

    def get_connected_apps(self) -> List[str]:
        """Get list of successfully connected apps"""
        return [url for url, conn in self._connections.items() if conn.connected]

    def is_connected(self, app_url: str) -> bool:
        """Check if specific app is connected"""
        normalized_url = app_url.rstrip('/')
        return normalized_url in self._connections and self._connections[normalized_url].connected