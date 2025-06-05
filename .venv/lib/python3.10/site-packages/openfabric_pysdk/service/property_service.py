import os
from typing import List, Union

from openfabric_pysdk.service import EnvironmentService, RemoteService
from openfabric_pysdk.store import KeyValueDB
from openfabric_pysdk.transport.schema.property_schema import Property, PropertySchema

database: KeyValueDB = KeyValueDB(f"properties", path=f"{os.getcwd()}/config", autodump=True)


#######################################################
#  Property service
#######################################################
class PropertyService:

    # ------------------------------------------------------------------------
    @staticmethod
    def get(name: str) -> Union[str, None]:
        options: List[Property] = PropertyService.get_all()
        if options is False or options is None or len(options) == 0:
            return None

        prop = next((option for option in options if option.name == name), Property())
        return prop.value

    # ------------------------------------------------------------------------
    @staticmethod
    def get_all() -> List[Property]:

        dos_node_url = EnvironmentService.get("DOS_CONNECTION")
        is_set = dos_node_url is not None and dos_node_url != ""

        if is_set:

            def deserializer(data):
                return PropertySchema().load(data, many=True)

            properties: List[Property] = RemoteService.get(f"{dos_node_url}/api/v1/appProperty", deserializer)
            if len(properties) != 0:
                return properties

        return PropertyService.get_default()

    # ------------------------------------------------------------------------
    @staticmethod
    def get_default() -> List[Property]:
        kvdb: List = database.all()
        if kvdb is False or kvdb is None or len(kvdb) == 0:
            return []

        return [Property(name=prop['name'], value=prop['value']) for prop in kvdb]
