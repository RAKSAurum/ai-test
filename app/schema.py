"""
Schema definitions for AI Pipeline Input/Output

This module defines the data structures used for communication with the AI pipeline.
Provides type-safe input and output schemas for the text-to-image-to-3D generation workflow.
"""

from openfabric_pysdk.loader import ConfigClass
from typing import Any, Optional


class Input(ConfigClass):
    """
    Input schema for the AI pipeline request.
    
    Defines the structure for incoming requests to the text-to-image-to-3D pipeline.
    Contains user prompt and optional session management parameters.
    
    Attributes:
        prompt (str): Text description for image and 3D model generation
        user_id (str): Identifier for the requesting user (defaults to "default")
        session_id (Optional[str]): Optional session identifier for tracking
    """
    
    def __init__(self):
        """Initialize input schema with default values."""
        super().__init__()
        self.prompt: str = ""
        self.user_id: str = "default"
        self.session_id: Optional[str] = None


class Output(ConfigClass):
    """
    Output schema for the AI pipeline response.
    
    Defines the structure for responses from the text-to-image-to-3D pipeline.
    Contains generation results, status information, and metadata.
    
    Attributes:
        result (Any): Main result data from the pipeline execution
        status (str): Execution status indicator (defaults to "success")
        message (str): Human-readable result description and file information
        generation_id (Optional[str]): Unique identifier for the completed generation
    """
    
    def __init__(self):
        """Initialize output schema with default values."""
        super().__init__()
        self.result: Any = None
        self.status: str = "success"
        self.message: str = ""
        self.generation_id: Optional[str] = None