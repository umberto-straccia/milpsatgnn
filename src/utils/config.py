import yaml

class Config(dict):
    """Configuration class that loads parameters from a YAML file.

    This class extends the built-in dict class to provide attribute-style access
    to configuration parameters loaded from a YAML file. It flattens nested dictionaries
    by concatenating keys with underscores.

    Args:
        config_path (str): Path to the YAML configuration file.

    Examples:
        >>> config = Config("config.yaml")
        >>> # If config.yaml contains: {"model": {"lr": 0.001}}
        >>> print(config.model_lr)
        0.001
        >>> print(config["model"]["lr"]) 
        0.001

    Methods:
        get(key, default=None): Safely access nested dictionary values using dot notation.
            Args:
                key (str): Period-separated string indicating nested keys (e.g. "model.lr")
                default: Value to return if key is not found
            Returns:
                Value if key exists, default otherwise
    """
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        super().__init__(config_dict)
        self._set_attrs(config_dict)

    def _set_attrs(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    setattr(self, f"{k.lower()}_{sub_k.lower()}", sub_v)
            else:
                setattr(self, k.lower(), v)

    def get(self, key, default=None):
        try:
            keys = key.split('.')
            val = self
            for k in keys:
                val = val[k]
            return val
        except:
            return default
       
def load_config(config):
    """
    Load and validate configuration parameters from a configuration object.
    This function takes a configuration object and creates a Config instance with the
    provided parameters.
    Args:
        config: A configuration object containing parameters for the model/application.
                The exact structure depends on the Config class implementation.
    Returns:
        Config: A validated Config instance containing all configuration parameters.
    Example:
        config_dict = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        config = load_config(config_dict)
    """

    return Config(config)