"""
Generic Registry for ML components.

Provides type-safe, thread-safe registration with:
- Decorator and explicit registration patterns
- Alias support for backward compatibility
- Legacy string import fallback
- Component metadata and filtering
"""

from typing import TypeVar, Generic, Dict, Type, Optional, Callable, List, Any
import importlib
import logging
import threading
import inspect

T = TypeVar('T')


class Registry(Generic[T]):
    """
    Type-safe component registry with decorator + explicit registration.

    Features:
        - Thread-safe registration
        - Alias support (multiple names for same component)
        - Legacy fallback to string-based imports
        - Metadata storage for filtering/discovery
        - Config validation against component signatures

    Usage:
        # Decorator registration
        @MODEL_REGISTRY.register('my_model', aliases=['mm'], tags=['transformer'])
        class MyModel(nn.Module):
            ...

        # Explicit registration
        MODEL_REGISTRY.register_class('other_model', OtherModel)

        # Retrieval (all equivalent)
        cls = MODEL_REGISTRY.get('my_model')
        cls = MODEL_REGISTRY.get('mm')  # alias
        cls = MODEL_REGISTRY.get('Models.path.MyModel')  # legacy fallback
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type[T]] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f'fusionlib.registry.{name}')

    @property
    def name(self) -> str:
        return self._name

    def register(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator for component registration.

        Args:
            name: Primary registration name
            aliases: Alternative names (for backward compat)
            description: Human-readable description
            tags: Searchable tags for filtering

        Returns:
            Decorator function
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self.register_class(
                name=name,
                cls=cls,
                aliases=aliases,
                description=description,
                tags=tags
            )
            return cls
        return decorator

    def register_class(
        self,
        name: str,
        cls: Type[T],
        aliases: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Explicitly register a class.

        Args:
            name: Primary registration name
            cls: Class to register
            aliases: Alternative names
            description: Human-readable description
            tags: Searchable tags
        """
        with self._lock:
            if name in self._registry:
                self._logger.warning(f"Overwriting '{name}' in {self._name} registry")

            self._registry[name] = cls
            self._metadata[name] = {
                'description': description or cls.__doc__ or "",
                'tags': tags or [],
                'module': cls.__module__,
                'class_name': cls.__name__,
            }

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases and self._aliases[alias] != name:
                        self._logger.warning(
                            f"Alias '{alias}' remapped: {self._aliases[alias]} -> {name}"
                        )
                    self._aliases[alias] = name

            self._logger.debug(f"Registered '{name}' -> {cls.__module__}.{cls.__name__}")

    def get(self, name: str, strict: bool = True) -> Type[T]:
        """
        Retrieve component by name, alias, or legacy path.

        Resolution order:
            1. Direct registry lookup
            2. Alias lookup
            3. Legacy string import (Models.xxx.ClassName)

        Args:
            name: Component name, alias, or import path
            strict: If True, raise KeyError on failure; else return None

        Returns:
            Component class

        Raises:
            KeyError: If component not found and strict=True
        """
        with self._lock:
            # Direct lookup
            if name in self._registry:
                return self._registry[name]

            # Alias lookup
            if name in self._aliases:
                canonical = self._aliases[name]
                return self._registry[canonical]

        # Legacy fallback: string-based import
        if '.' in name:
            try:
                return self._import_class(name)
            except (ImportError, AttributeError) as e:
                if strict:
                    raise KeyError(
                        f"'{name}' not found in {self._name} registry and import failed: {e}\n"
                        f"Available: {self.list_available()}"
                    )
                return None

        if strict:
            raise KeyError(
                f"'{name}' not found in {self._name} registry.\n"
                f"Available: {self.list_available()}\n"
                f"Aliases: {list(self._aliases.keys())}"
            )
        return None

    def _import_class(self, import_str: str) -> Type[T]:
        """Import class from dotted string path."""
        mod_str, _, class_str = import_str.rpartition('.')
        module = importlib.import_module(mod_str)
        return getattr(module, class_str)

    def list_available(self, tags: Optional[List[str]] = None) -> List[str]:
        """
        List registered component names.

        Args:
            tags: If provided, filter by tags (AND logic)

        Returns:
            List of component names
        """
        with self._lock:
            if not tags:
                return sorted(self._registry.keys())

            # Filter by tags
            result = []
            for name, meta in self._metadata.items():
                if all(t in meta.get('tags', []) for t in tags):
                    result.append(name)
            return sorted(result)

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered component."""
        with self._lock:
            if name in self._aliases:
                name = self._aliases[name]
            return self._metadata.get(name)

    def get_signature(self, name: str) -> Optional[inspect.Signature]:
        """Get __init__ signature for a component."""
        cls = self.get(name, strict=False)
        if cls is None:
            return None
        return inspect.signature(cls.__init__)

    def validate_config(self, name: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate config against component's __init__ signature.

        Args:
            name: Component name
            config: Config dict to validate

        Returns:
            List of validation errors (empty if valid)
        """
        sig = self.get_signature(name)
        if sig is None:
            return [f"Component '{name}' not found"]

        errors = []
        params = sig.parameters

        for key in config:
            if key not in params and 'kwargs' not in str(params):
                errors.append(f"Unknown parameter: '{key}'")

        for param_name, param in params.items():
            if param_name == 'self':
                continue
            if param.default == inspect.Parameter.empty and param_name not in config:
                if param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD
                ):
                    errors.append(f"Missing required parameter: '{param_name}'")

        return errors

    def __contains__(self, name: str) -> bool:
        return name in self._registry or name in self._aliases

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry('{self._name}', {len(self._registry)} components)"


# Global registry instances
MODEL_REGISTRY: Registry = Registry('models')
ENCODER_REGISTRY: Registry = Registry('encoders')
LOSS_REGISTRY: Registry = Registry('losses')
