import yaml
from transformers import pipeline

def load_model_config(config_path: str) -> dict:
    """
    Loads the model configuration from a YAML file.
    
    Parameters:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def create_pipeline(config: dict):
    """
    Creates a text generation pipeline using the given configuration.
    
    Parameters:
        config (dict): Configuration dictionary with keys for pipeline setup.
        
    Returns:
        transformers.Pipeline: The text generation pipeline.
    """
    pipeline_config = config.get("pipeline", {})
    task = pipeline_config.get("task", "text-generation")
    model_name = pipeline_config.get("model_name", "EleutherAI/gpt-neo-1.3B")
    return pipeline(task, model=model_name)

if __name__ == "__main__":
    config_path = "/Users/blairjdaniel/AI-Assistant-Springs/config/model_config.yaml"
    config = load_model_config(config_path)
    pipe = create_pipeline(config)
    print("Pipeline created with model:", config["pipeline"]["model_name"])