"""
Configuration management for HERMES pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class OCRConfig(BaseModel):
    """OCR configuration."""
    enabled: bool = True
    min_chars_per_page: int = 100
    dpi: int = 300
    language: str = "eng"


class LoaderConfig(BaseModel):
    """Loader configuration."""
    use_docling: bool = False
    extract_images: bool = True
    extract_tables: bool = True
    ocr_fallback: bool = True


class EmbeddingModelConfig(BaseModel):
    """Individual embedding model configuration."""
    model_name: Optional[str] = None
    truncate_dim: Optional[int] = None
    dimension: Optional[int] = None
    adapter_mask: Optional[str] = None
    batch_size: int = 32
    max_length: int = 8192


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    primary_model: str = "jina_v4"
    models: Dict[str, EmbeddingModelConfig] = Field(default_factory=dict)


class DSPyConfig(BaseModel):
    """DSPy configuration."""
    enabled: bool = True
    model: str = "gpt-4"
    training_examples_path: str = "./training_examples"
    optimize_on_load: bool = False


class ConveyanceConfig(BaseModel):
    """Conveyance analysis configuration."""
    analyze_algorithms: bool = True
    analyze_equations: bool = True
    analyze_examples: bool = True
    analyze_code: bool = True
    complexity_threshold: float = 0.7


class StorageConfig(BaseModel):
    """Storage configuration."""
    backend: str = "arangodb"
    connection: Dict[str, Any] = Field(default_factory=dict)
    graph: Dict[str, Any] = Field(default_factory=dict)
    edges: Dict[str, Any] = Field(default_factory=dict)


class ISNEConfig(BaseModel):
    """ISNE configuration."""
    enabled: bool = True
    embedding_dim: int = 128
    training_iterations: int = 100
    use_adaptive: bool = True


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    batch_size: int = 10
    max_workers: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    progress_bar: bool = True


class ProcessingConfig(BaseModel):
    """Document processing configuration."""
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    extraction: Dict[str, bool] = Field(default_factory=lambda: {
        "clean_text": True,
        "preserve_structure": True,
        "extract_metadata": True
    })


class MetadataConfig(BaseModel):
    """Metadata extraction configuration."""
    dspy: DSPyConfig = Field(default_factory=DSPyConfig)
    conveyance: ConveyanceConfig = Field(default_factory=ConveyanceConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "hermes.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class HermesConfig(BaseModel):
    """Complete HERMES configuration."""
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    loaders: Dict[str, LoaderConfig] = Field(default_factory=dict)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    isne: ISNEConfig = Field(default_factory=ISNEConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @validator('storage', pre=True)
    def expand_env_vars(cls, v):
        """Expand environment variables in storage config."""
        if isinstance(v, dict) and 'connection' in v:
            conn = v['connection']
            for key, value in conn.items():
                if isinstance(value, str) and value.startswith('${'):
                    # Handle ${VAR:-default} syntax
                    if ':-' in value:
                        var_name, default = value[2:-1].split(':-', 1)
                        conn[key] = os.getenv(var_name, default)
                    else:
                        var_name = value[2:-1]
                        conn[key] = os.getenv(var_name, '')
        return v


def load_config(config_path: Optional[Path] = None) -> HermesConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for:
                    1. HERMES_CONFIG env var
                    2. config.yaml in current directory
                    3. config.yaml in home directory
                    4. Uses defaults
                    
    Returns:
        HermesConfig object
    """
    # Find config file
    if config_path is None:
        # Check environment variable
        env_path = os.getenv('HERMES_CONFIG')
        if env_path:
            config_path = Path(env_path)
        else:
            # Check standard locations
            for path in [
                Path('config.yaml'),
                Path('hermes_config.yaml'),
                Path.home() / '.hermes' / 'config.yaml',
            ]:
                if path.exists():
                    config_path = path
                    break
    
    # Load config if found
    if config_path and config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Handle nested hermes_config key
            if 'hermes_config' in config_dict:
                config_dict = config_dict['hermes_config']
                
            return HermesConfig(**config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return HermesConfig()
    else:
        logger.info("No config file found, using defaults")
        return HermesConfig()


def save_config(config: HermesConfig, config_path: Path):
    """
    Save configuration to YAML file.
    
    Args:
        config: HermesConfig object
        config_path: Path to save config
    """
    config_dict = {'hermes_config': config.dict()}
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
    logger.info(f"Configuration saved to {config_path}")


def setup_logging(config: HermesConfig):
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger('hermes').setLevel(getattr(logging, config.logging.level.upper()))