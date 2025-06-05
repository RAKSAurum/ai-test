import json
import logging
import re
from typing import Any, Union

from openfabric_pysdk.flask import *


#######################################################
#  Remote service
#######################################################
class RemoteService:

    # ------------------------------------------------------------------------
    @staticmethod
    def get(url: str, deserializer=None):
        return RemoteService.__unwrap(get(RemoteService.__normalize(url)), deserializer=deserializer)

    # ------------------------------------------------------------------------
    @staticmethod
    def post(url: str, data: Any, deserializer=None):
        return RemoteService.__unwrap(post(RemoteService.__normalize(url), json=data), deserializer=deserializer)

    # ------------------------------------------------------------------------
    @staticmethod
    def patch(url: str, data: Any, deserializer=None):
        return RemoteService.__unwrap(patch(RemoteService.__normalize(url), json=data), deserializer=deserializer)

    # ------------------------------------------------------------------------
    @staticmethod
    def delete(url: str, deserializer=None):
        return RemoteService.__unwrap(delete(RemoteService.__normalize(url)), deserializer=deserializer)

    # ------------------------------------------------------------------------
    @staticmethod
    def __normalize(url) -> Union[str, None]:
        if url is None:
            return None
        split_url = re.split('://', url, 1)
        if len(split_url) == 2:
            protocol, rest = split_url
            normalized_rest = re.sub(r'/{2,}', '/', rest)
            return f'{protocol}://{normalized_rest}'
        return url

    # ------------------------------------------------------------------------
    @staticmethod
    def __unwrap(response: Response, deserializer=None):
        if response.ok is False:
            logging.error(f"Response code: {response.status_code} on url: {response.url}")
            return None
        json_object = json.loads(response.text)

        if deserializer is None:
            return json_object

        return deserializer(json_object)
