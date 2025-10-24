from pytket.extensions.quantinuum import QuantinuumBackend


class QuantinuumWrapper:
    _instance = None
    __local = True

    def __new__(cls, target: QuantinuumBackend = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, target: QuantinuumBackend = None):
        if not self._initialized:
            self._target = target
            self._initialized = True
            self.__local = False

    @classmethod
    def get_target(cls):
        if cls.__local is True:
            from pytket.extensions.quantinuum import QuantinuumBackend
            from pytket.extensions.quantinuum import QuantinuumAPIOffline
            api_offline = QuantinuumAPIOffline()
            return QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError("Quantinuum not initialized.")
        return cls._instance._target