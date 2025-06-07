"""
Remote API Connection Stub

This module provides a simplified interface for connecting to and calling remote applications
via HTTP API endpoints. It handles connection management and API calls with proper error handling.
"""

import requests
import logging
from typing import List, Dict
import json


class Remote:
    """
    Represents a connection to a remote application.
    
    Handles the connection lifecycle and API calls to a single remote endpoint.
    """
    
    def __init__(self, url: str, proxy_name: str):
        """
        Initialize a remote connection.
        
        Args:
            url: The base URL of the remote application
            proxy_name: Name identifier for this proxy connection
        """
        self.url = url
        self.proxy_name = proxy_name
        self.connected = False

    def connect(self):
        """
        Establish connection to the remote application by checking its manifest endpoint.
        
        Returns:
            self: Returns self for method chaining
        """
        try:
            response = requests.get(f"{self.url}/manifest", timeout=10)
            self.connected = response.status_code == 200
            if self.connected:
                logging.info(f"✅ Connected to {self.url}")
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            self.connected = False
        return self

    def call(self, input_data: dict):
        """
        Make an API call to the remote application.
        
        Args:
            input_data: Dictionary containing the data to send to the remote app
            
        Returns:
            dict: Response from the remote application
            
        Raises:
            Exception: If not connected or if the API call fails
        """
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
                logging.info("✅ API call successful")
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
    """
    Main stub class for managing connections to multiple remote applications.
    
    Provides a unified interface for connecting to and calling multiple remote apps,
    with fallback mechanisms for compatibility.
    """
    
    def __init__(self, app_urls: List[str]):
        """
        Initialize the stub with a list of application URLs.
        
        Args:
            app_urls: List of URLs for remote applications to connect to
        """
        self._connections: Dict[str, Remote] = {}
        self.apps = app_urls
        self._establish_connections(app_urls)
        self._log_connection_status()

    def _normalize_url(self, url: str) -> str:
        """
        Normalize a URL to ensure consistent formatting.
        
        Args:
            url: Raw URL string
            
        Returns:
            str: Normalized URL with https protocol and no trailing slash
        """
        url = url.strip('/').replace('http://', 'https://')
        if not url.startswith('https://'):
            url = f"https://{url}"
        return url.rstrip('/')

    def _establish_connections(self, app_urls: List[str]):
        """
        Attempt to establish connections to all provided application URLs.
        
        Args:
            app_urls: List of application URLs to connect to
        """
        for app_url in app_urls:
            normalized_url = self._normalize_url(app_url)
            
            try:
                # Test connection via manifest endpoint
                manifest_response = requests.get(f"{normalized_url}/manifest", timeout=10)
                if manifest_response.status_code == 200:
                    remote = Remote(normalized_url, f"{normalized_url}-proxy")
                    self._connections[normalized_url] = remote.connect()
                    if remote.connected:
                        logging.info(f"🎯 Connected to {normalized_url}")
            except Exception as e:
                logging.error(f"❌ Failed to connect to {normalized_url}: {e}")

    def _log_connection_status(self):
        """Log the overall connection status after initialization."""
        connected_apps = self.get_connected_apps()
        if connected_apps:
            logging.info(f"✅ Connected to {len(connected_apps)} apps")
        else:
            logging.warning("⚠️ No apps connected - using direct API calls")

    def call(self, app_url: str, input_data: dict, user_id: str) -> dict:
        """
        Call a remote application with input data.
        
        This is a compatibility wrapper that attempts to use established connections
        but falls back gracefully for direct API usage.
        
        Args:
            app_url: URL of the application to call
            input_data: Data to send to the application
            user_id: User identifier (maintained for compatibility)
            
        Returns:
            dict: Response from the remote application or fallback indicator
            
        Raises:
            Exception: If the API call fails
        """
        normalized_url = self._normalize_url(app_url)
        
        # Check if we have an established connection
        if normalized_url not in self._connections:
            logging.warning(f"Connection not found for: {normalized_url}, using direct calls")
            return {"result": "using_direct_api_calls"}
        
        connection = self._connections[normalized_url]
        if not connection.connected:
            logging.warning(f"Not connected to: {normalized_url}, using direct calls")
            return {"result": "using_direct_api_calls"}
        
        try:
            return connection.call(input_data)
        except Exception as e:
            logging.error(f"❌ Call failed: {e}")
            raise

    def get_connected_apps(self) -> List[str]:
        """
        Get a list of successfully connected application URLs.
        
        Returns:
            List[str]: URLs of applications that are currently connected
        """
        return [url for url, conn in self._connections.items() if conn.connected]

    def is_connected(self, app_url: str) -> bool:
        """
        Check if a specific application is connected.
        
        Args:
            app_url: URL of the application to check
            
        Returns:
            bool: True if the application is connected, False otherwise
        """
        normalized_url = self._normalize_url(app_url)
        return (normalized_url in self._connections and 
                self._connections[normalized_url].connected)