import yaml


class Config:
    _instance = None
    data = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            with open("config.yaml", "r") as file:
                cls.data = yaml.safe_load(file)
        return cls._instance
