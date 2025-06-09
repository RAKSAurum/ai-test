"""
Schema definitions for AI Pipeline Input/Output

This module defines the data structures used for communication with the AI pipeline.
Provides type-safe input and output schemas for the text-to-image-to-3D generation workflow.
"""

from typing import Any, Optional

from openfabric_pysdk.loader import ConfigClass


class Input(ConfigClass):
    """
    Input schema for the AI pipeline request.
    
    Defines the structure for incoming requests to the text-to-image-to-3D pipeline.
    Contains user prompt and optional session management parameters for tracking
    user interactions and maintaining conversational context.
    
    Attributes:
        prompt (str): Text description for image and 3D model generation.
        user_id (str): Identifier for the requesting user (defaults to "default").
        session_id (Optional[str]): Optional session identifier for tracking
            user sessions and maintaining conversation context.
    
    Example:
        >>> input_data = Input()
        >>> input_data.prompt = "A futuristic robot with glowing blue eyes"
        >>> input_data.user_id = "user123"
    """
    
    def __init__(self) -> None:
        """Initialize input schema with default values."""
        super().__init__()
        self.prompt: str = ""
        self.user_id: str = "default"
        self.session_id: Optional[str] = None


class Output(ConfigClass):
    """
    Output schema for the AI pipeline response.
    
    Defines the structure for responses from the text-to-image-to-3D pipeline.
    Contains generation results, status information, and metadata for tracking
    pipeline execution and providing detailed feedback to users.
    
    Attributes:
        result (Any): Main result data from the pipeline execution, typically
            containing file paths or generation metadata.
        status (str): Execution status indicator (defaults to "success").
            Common values include "success", "error", "processing".
        message (str): Human-readable result description and file information,
            including paths to generated images and 3D models.
        generation_id (Optional[str]): Unique identifier for the completed generation,
            used for tracking and memory storage purposes.
    
    Example:
        >>> output = Output()
        >>> output.status = "success"
        >>> output.message = "Generated robot model successfully"
        >>> output.generation_id = "gen_12345"
    """
    
    def __init__(self) -> None:
        """Initialize output schema with default values."""
        super().__init__()
        self.result: Any = None
        self.status: str = "success"
        self.message: str = ""
        self.generation_id: Optional[str] = None