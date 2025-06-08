"""
Remote API Connection Stub

This module provides a simplified interface for connecting to and calling remote applications
via HTTP API endpoints. It handles connection management and API calls with proper error handling.
"""

import json
import logging
from typing import Dict, List

import requests


class Remote:
    """
    Represents a connection to a remote application.
    
    Handles the connection lifecycle and API calls to a single remote endpoint.
    This class manages individual connections and provides methods for testing
    connectivity and making API calls.
    
    Attributes:
        url (str): The base URL of the remote application.
        proxy_name (str): Name identifier for this proxy connection.
        connected (bool): Current connection status.
    
    Example:
        >>> remote = Remote("https://api.example.com", "example-proxy")
        >>> remote.connect()
        >>> response = remote.call({"input": "data"})
    """
    
    def __init__(self, url: str, proxy_name: str) -> None:
        """
        Initialize a remote connection.
        
        Args:
            url (str): The base URL of the remote application.
            proxy_name (str): Name identifier for this proxy connection.
        """
        self.url = url
        self.proxy_name = proxy_name
        self.connected = False

    def connect(self) -> 'Remote':
        """
        Establish connection to the remote application.
        
        Tests connectivity by checking the manifest endpoint of the remote application.
        Updates the connection status based on the response.
        
        Returns:
            Remote: Returns self for method chaining.
            
        Note:
            Uses a 10-second timeout for the connection test.
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

    def call(self, input_data: dict) -> dict:
        """
        Make an API call to the remote application.
        
        Sends a POST request to the execution endpoint with the provided input data.
        Handles JSON parsing and provides fallback for non-JSON responses.
        
        Args:
            input_data (dict): Dictionary containing the data to send to the remote app.
            
        Returns:
            dict: Response from the remote application, either parsed JSON or
                wrapped text response.
            
        Raises:
            Exception: If not connected or if the API call fails.
            
        Note:
            Uses a 60-second timeout for API calls to accommodate longer processing times.
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
                    # Fallback for non-JSON responses
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
    with fallback mechanisms for compatibility. This class handles connection pooling,
    URL normalization, and provides methods for checking connection status.
    
    Attributes:
        _connections (Dict[str, Remote]): Internal mapping of URLs to Remote instances.
        apps (List[str]): List of application URLs provided during initialization.
    
    Example:
        >>> stub = Stub(["https://api1.example.com", "https://api2.example.com"])
        >>> response = stub.call("https://api1.example.com", {"input": "data"}, "user123")
        >>> connected_apps = stub.get_connected_apps()
    """
    
    def __init__(self, app_urls: List[str]) -> None:
        """
        Initialize the stub with a list of application URLs.
        
        Automatically attempts to establish connections to all provided URLs
        and logs the overall connection status.
        
        Args:
            app_urls (List[str]): List of URLs for remote applications to connect to.
        """
        self._connections: Dict[str, Remote] = {}
        self.apps = app_urls
        self._establish_connections(app_urls)
        self._log_connection_status()

    def _normalize_url(self, url: str) -> str:
        """
        Normalize a URL to ensure consistent formatting.
        
        Converts HTTP to HTTPS, adds protocol if missing, and removes trailing slashes
        for consistent URL handling across the application.
        
        Args:
            url (str): Raw URL string that may need normalization.
            
        Returns:
            str: Normalized URL with https protocol and no trailing slash.
            
        Example:
            >>> stub._normalize_url("http://example.com/")
            "https://example.com"
        """
        url = url.strip('/').replace('http://', 'https://')
        if not url.startswith('https://'):
            url = f"https://{url}"
        return url.rstrip('/')

    def _establish_connections(self, app_urls: List[str]) -> None:
        """
        Attempt to establish connections to all provided application URLs.
        
        Tests each URL by hitting the manifest endpoint and creates Remote instances
        for successful connections. Failed connections are logged but don't prevent
        initialization.
        
        Args:
            app_urls (List[str]): List of application URLs to connect to.
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

    def _log_connection_status(self) -> None:
        """
        Log the overall connection status after initialization.
        
        Provides summary information about successful connections or warnings
        if no connections were established.
        """
        connected_apps = self.get_connected_apps()
        if connected_apps:
            logging.info(f"✅ Connected to {len(connected_apps)} apps")
        else:
            logging.warning("⚠️ No apps connected - using direct API calls")

    def call(self, app_url: str, input_data: dict, user_id: str) -> dict:
        """
        Call a remote application with input data.
        
        This is a compatibility wrapper that attempts to use established connections
        but falls back gracefully for direct API usage. The user_id parameter is
        maintained for API compatibility but not currently used in the implementation.
        
        Args:
            app_url (str): URL of the application to call.
            input_data (dict): Data to send to the application.
            user_id (str): User identifier (maintained for compatibility).
            
        Returns:
            dict: Response from the remote application or fallback indicator
                containing {"result": "using_direct_api_calls"} if no connection exists.
            
        Raises:
            Exception: If the API call fails after a successful connection.
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
            List[str]: URLs of applications that are currently connected and available
                for API calls.
        """
        return [url for url, conn in self._connections.items() if conn.connected]

    def is_connected(self, app_url: str) -> bool:
        """
        Check if a specific application is connected.
        
        Args:
            app_url (str): URL of the application to check.
            
        Returns:
            bool: True if the application is connected and ready for API calls,
                False otherwise.
        """
        normalized_url = self._normalize_url(app_url)
        return (normalized_url in self._connections and 
                self._connections[normalized_url].connected)