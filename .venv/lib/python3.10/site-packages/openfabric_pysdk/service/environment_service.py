import os


# ################################################################
# #  Environment Service
# ################################################################
class EnvironmentService:

    # ----------------------------------------------------------------
    @staticmethod
    def get(key: str) -> str:
        return os.getenv(key)
