from typing import Optional, Union

from openfabric_pysdk.helper import Proxy
from openfabric_pysdk.helper.proxy import ExecutionResult


class Remote:
    """
    A helper class for interfacing with Openfabric Proxy instances.
    
    This class provides a simplified interface for connecting to and executing
    requests through an Openfabric Proxy service, supporting both synchronous
    and asynchronous execution patterns.
    
    Attributes:
        proxy_url (str): The URL to the proxy service.
        proxy_tag (Optional[str]): An optional tag to identify a specific proxy instance.
        client (Optional[Proxy]): The initialized proxy client instance.
    
    Example:
        >>> remote = Remote("https://api.example.com")
        >>> remote.connect()
        >>> result = remote.execute_sync({"input": "data"}, {}, "unique_id")
    """

    def __init__(self, proxy_url: str, proxy_tag: Optional[str] = None) -> None:
        """
        Initialize the Remote instance with proxy configuration.

        Args:
            proxy_url (str): The base URL of the proxy service.
            proxy_tag (Optional[str], optional): An optional tag for identifying
                a specific proxy instance. Defaults to None.
        """
        self.proxy_url = proxy_url
        self.proxy_tag = proxy_tag
        self.client: Optional[Proxy] = None

    def connect(self) -> 'Remote':
        """
        Establish a connection with the proxy service.
        
        Creates and initializes the Proxy client instance with SSL verification
        disabled for development environments.

        Returns:
            Remote: The current instance to enable method chaining.
            
        Note:
            SSL verification is disabled (ssl_verify=False) which should be
            reconsidered for production environments.
        """
        self.client = Proxy(self.proxy_url, self.proxy_tag, ssl_verify=False)
        return self

    def execute(self, inputs: dict, uid: str) -> Union[ExecutionResult, None]:
        """
        Execute an asynchronous request through the proxy client.

        Args:
            inputs (dict): The input payload to send to the proxy service.
            uid (str): A unique identifier for tracking the request.

        Returns:
            Union[ExecutionResult, None]: The execution result object for
                asynchronous processing, or None if client is not connected.
                
        Note:
            This method returns immediately without waiting for completion.
            Use get_response() to wait for and process the result.
        """
        if self.client is None:
            return None

        return self.client.request(inputs, uid)

    @staticmethod
    def get_response(output: ExecutionResult) -> Union[dict, None]:
        """
        Wait for execution completion and extract the response data.

        Args:
            output (ExecutionResult): The result object returned from a proxy request.

        Returns:
            Union[dict, None]: The response data if execution completed successfully,
                None if output is None or execution is still pending.

        Raises:
            Exception: If the request failed or was cancelled.
            
        Note:
            This method blocks until the execution completes.
        """
        if output is None:
            return None

        # Wait for the execution to complete
        output.wait()
        
        # Check execution status
        status = str(output.status()).lower()
        if status == "completed":
            return output.data()
        elif status in ("cancelled", "failed"):
            raise Exception("The request to the proxy app failed or was cancelled!")
        
        return None

    def execute_sync(self, inputs: dict, configs: dict, uid: str) -> Union[dict, None]:
        """
        Execute a synchronous request with configuration parameters.
        
        This method combines execution and response processing into a single
        blocking call for simplified usage patterns.

        Args:
            inputs (dict): The input payload to send to the proxy.
            configs (dict): Additional configuration parameters for the execution.
            uid (str): A unique identifier for tracking the request.

        Returns:
            Union[dict, None]: The processed response data if successful,
                None if client is not connected or execution failed.
                
        Raises:
            Exception: If the request failed or was cancelled (propagated from get_response).
        """
        if self.client is None:
            return None

        # Execute the request and wait for completion
        output = self.client.execute(inputs, configs, uid)
        return Remote.get_response(output)