import json
import logging
import requests
from typing import Any, Dict, List, Literal, Tuple

from core.remote import Remote
from openfabric_pysdk.utility import LoaderUtil
from openfabric_pysdk.context import State, Ray
from openfabric_pysdk.loader import ConfigClass

# Type aliases for clarity
Manifests = Dict[str, dict]
Schemas = Dict[str, Tuple[dict, dict]]
Connections = Dict[str, Remote]

class Stub:
    """
    Stub acts as a lightweight client interface that initializes remote connections
    to multiple Openfabric applications, fetching their manifests, schemas, and enabling
    execution of calls to these apps.
    """

    def __init__(self, app_ids: List[str]):
        """Initialize the Stub with Openfabric app IDs"""
        self._schema: Schemas = {}
        self._manifest: Manifests = {}
        self._connections: Connections = {}

        for app_id in app_ids:
            base_url = app_id.strip('/')
            try:
                # Fetch manifest
                manifest = requests.get(f"https://{base_url}/manifest", timeout=10).json()
                logging.info(f"[{app_id}] Manifest loaded successfully")
                self._manifest[app_id] = manifest

                # Fetch schemas
                input_schema = requests.get(f"https://{base_url}/schema?type=input", timeout=10).json()
                output_schema = requests.get(f"https://{base_url}/schema?type=output", timeout=10).json()
                logging.info(f"[{app_id}] Schemas loaded successfully")
                self._schema[app_id] = (input_schema, output_schema)

                # Establish WebSocket connection
                self._connections[app_id] = Remote(f"wss://{base_url}/app", f"{app_id}-proxy").connect()
                logging.info(f"[{app_id}] Connection established successfully")
                
            except Exception as e:
                logging.error(f"[{app_id}] Initialization failed: {e}")

    def call(self, app_id: str, data: Any, uid: str = 'super-user') -> dict:
        """Execute a call to the specified Openfabric app"""
        connection = self._connections.get(app_id)
        if not connection:
            raise Exception(f"Connection not found for app ID: {app_id}")

        try:
            handler = connection.execute(data, uid)
            result = connection.get_response(handler)
            
            # Basic resource handling without deprecated functions
            if isinstance(result, dict) and 'result' in result:
                result_data = result['result']
                # Handle resource URLs if they exist
                if isinstance(result_data, str) and result_data.startswith('resource://'):
                    resource_id = result_data.replace('resource://', '')
                    resource_url = f"https://{app_id}/resource?reid={resource_id}"
                    try:
                        response = requests.get(resource_url, timeout=15)
                        if response.status_code == 200:
                            result['result'] = response.content
                    except Exception as e:
                        logging.warning(f"Failed to resolve resource {resource_id}: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"[{app_id}] Execution failed: {e}")
            raise e

    def manifest(self, app_id: str) -> dict:
        """Get manifest for specified app"""
        return self._manifest.get(app_id, {})

    def schema(self, app_id: str, type: Literal['input', 'output']) -> dict:
        """Get input or output schema for specified app"""
        _input, _output = self._schema.get(app_id, (None, None))
        
        if type == 'input':
            if _input is None:
                raise ValueError(f"Input schema not found for app ID: {app_id}")
            return _input
        elif type == 'output':
            if _output is None:
                raise ValueError(f"Output schema not found for app ID: {app_id}")
            return _output
        else:
            raise ValueError("Type must be either 'input' or 'output'")