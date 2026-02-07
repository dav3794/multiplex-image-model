"""Registry system for blocks, stages, and encoders.

This module provides a centralized registry for different architecture components,
allowing flexible configuration through string identifiers.
"""

from typing import Callable

from .base_modules import Block, Encoder


class Registry:
    """Registry for architecture components."""

    def __init__(self, name: str):
        """Initialize a registry.

        Args:
            name: Name of the registry (e.g., 'block', 'encoder')
        """
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, name: str = None) -> Callable:
        """Register a class in the registry.

        Can be used as a decorator:
            @BLOCK_REGISTRY.register("convnext")
            class ConvNextBlock(Block):
                ...

        Or called directly:
            BLOCK_REGISTRY.register("convnext")(ConvNextBlock)

        Args:
            name: Name to register the class under. If None, uses class.__name__

        Returns:
            Decorator function
        """

        def decorator(cls: type) -> type:
            registry_name = name if name is not None else cls.__name__
            if registry_name in self._registry:
                raise ValueError(
                    f"'{registry_name}' is already registered in {self.name} registry"
                )
            self._registry[registry_name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        """Get a registered class by name.

        Args:
            name: Name of the registered class

        Returns:
            The registered class

        Raises:
            KeyError: If name is not found in registry
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. Available: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list:
        """List all available registered names.

        Returns:
            List of registered names
        """
        return list(self._registry.keys())


# Create global registries
BLOCK_REGISTRY = Registry("block")
ENCODER_REGISTRY = Registry("encoder")


def build_from_config(registry: Registry, config: dict):
    """Build an instance from a configuration dictionary.

    Args:
        registry: The registry to use for looking up the class
        config: Configuration dictionary with 'type' and optional 'module_parameters' keys

    Returns:
        Instance of the registered class

    Example:
        config = {
            "type": "convnext",
            "module_parameters": {
                "dim": 128,
                "kernel_size": 7
            }
        }
        block = build_from_config(BLOCK_REGISTRY, config)
    """
    if isinstance(config, str):
        # If config is just a string, use it as the type with no parameters
        config = {"type": config}

    if "type" not in config:
        raise ValueError(f"Config must contain 'type' key. Got: {config}")

    cls = registry.get(config["type"])
    module_parameters = config.get("module_parameters", {})

    return cls(**module_parameters)


def resolve_block_class(block_type: str | type[Block] | dict) -> type[Block]:
    """Resolve block type to a class.

    Args:
        block_type: Either a string (registry name), a Block class, or a config dict

    Returns:
        Block class
    """
    if isinstance(block_type, type) and issubclass(block_type, Block):
        return block_type
    elif isinstance(block_type, str):
        return BLOCK_REGISTRY.get(block_type)
    elif isinstance(block_type, dict):
        return BLOCK_REGISTRY.get(block_type["type"])
    else:
        raise ValueError(f"Invalid block_type: {block_type}")


def resolve_encoder_class(encoder_type: str | type[Encoder] | dict) -> type[Encoder]:
    """Resolve encoder type to a class.

    Args:
        encoder_type: Either a string (registry name), an Encoder class, or a config dict

    Returns:
        Encoder class
    """
    if isinstance(encoder_type, type) and issubclass(encoder_type, Encoder):
        return encoder_type
    elif isinstance(encoder_type, str):
        return ENCODER_REGISTRY.get(encoder_type)
    elif isinstance(encoder_type, dict):
        return ENCODER_REGISTRY.get(encoder_type["type"])
    else:
        raise ValueError(f"Invalid encoder_type: {encoder_type}")
