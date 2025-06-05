import json
import logging
import requests
from typing import Any, Dict, List, Literal, Tuple

from core.remote import Remote
from openfabric_pysdk.utility import LoaderUtil
from openfabric_pysdk.context import State, Ray
from openfabric_pysdk.loader import ConfigClass

class Stub:
    def __init__(self, app_ids: List[str]):
        self._schema: Dict[str, Tuple[dict, dict]] = {}
        self._manifest: Dict[str, dict] = {}
        self._connections: Dict[str, Remote] = {}

        for app_id in app_ids:
            base_url = app_id.strip('/')
            try:
                manifest = requests.get(f"https://{base_url}/manifest", timeout=10).json()
                input_schema = requests.get(f"https://{base_url}/schema?type=input", timeout=10).json()
                output_schema = requests.get(f"https://{base_url}/schema?type=output", timeout=10).json()
                
                self._manifest[app_id] = manifest
                self._schema[app_id] = (input_schema, output_schema)
                self._connections[app_id] = Remote(f"wss://{base_url}/app", f"{app_id}-proxy").connect()
                
                logging.info(f"[{app_id}] Successfully initialized")
            except Exception as e:
                logging.error(f"[{app_id}] Initialization failed: {e}")

    def call(self, app_id: str, data: Any, uid: str = 'super-user') -> dict:
        connection = self._connections.get(app_id)
        if not connection:
            raise Exception(f"Connection not found for app ID: {app_id}")
        
        handler = connection.execute(data, uid)
        result = connection.get_response(handler)
        
        # Handle resource URLs if present
        if isinstance(result, dict) and 'result' in result:
            result_data = result['result']
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

    def manifest(self, app_id: str) -> dict:
        return self._manifest.get(app_id, {})

    def schema(self, app_id: str, type: Literal['input', 'output']) -> dict:
        _input, _output = self._schema.get(app_id, (None, None))
        if type == 'input':
            return _input or {}
        elif type == 'output':
            return _output or {}
        else:
            raise ValueError("Type must be either 'input' or 'output'")