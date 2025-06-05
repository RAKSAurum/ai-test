from openfabric_pysdk.loader import ConfigClass
from typing import Any, Optional

class Input(ConfigClass):
    """Input schema for your AI pipeline"""
    def __init__(self):
        super().__init__()
        self.prompt: str = ""
        self.user_id: str = "default"
        self.session_id: Optional[str] = None

class Output(ConfigClass):
    """Output schema for your AI pipeline"""
    def __init__(self):
        super().__init__()
        self.result: Any = None
        self.status: str = "success"
        self.message: str = ""
        self.generation_id: Optional[str] = None