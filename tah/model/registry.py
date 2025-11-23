import inspect
from typing import Dict, Type, Callable, Optional, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from tah.model.iter_decider import IterDecider
    from tah.model.input_updater import InputUpdater
    from tah.model.output_updater import OutputUpdater
    from tah.model.iter_label import IterLabelGenerator
    from tah.model.loss import LossFunc

T = TypeVar('T')

def create_registry(registry_name: str, case_insensitive: bool = False) -> Tuple[Dict[str, Type[T]], Callable, Callable[[str], Type[T]]]:
    """
    Create a registry system with register and get functions.
    
    Args:
        registry_name: Name for error messages
        case_insensitive: Whether to store lowercase versions of names
    
    Returns:
        Tuple of (registry_dict, register_function, get_function)
    """
    registry: Dict[str, Type[T]] = {}
    
    def register(cls_or_name=None, name: Optional[str] = None):
        """Register a class in the registry. Supports multiple usage patterns."""
        def _register(c: Type[T]) -> Type[T]:
            # Determine the name to use
            if isinstance(cls_or_name, str):
                # Called as @register("name")
                class_name = cls_or_name
            elif name is not None:
                # Called as @register(name="name") or register(cls, name="name")
                class_name = name
            else:
                # Use class name
                class_name = c.__name__
            
            # Store in registry
            registry[class_name] = c
            if case_insensitive:
                registry[class_name.lower()] = c
            return c
        
        if cls_or_name is not None and not isinstance(cls_or_name, str):
            # Called as @register or register(cls)
            return _register(cls_or_name)
        else:
            # Called as @register("name") or @register(name="name")
            return _register
    
    def get_class(name: str) -> Type[T]:
        """Get class by name from registry."""
        key = name if name in registry else (name.lower() if case_insensitive else name)
        if key not in registry:
            available = list(registry.keys())
            raise ValueError(f"Unknown {registry_name} class: {name}. Available: {available}")
        return registry[name]
    
    return registry, register, get_class


# Create model registry (case insensitive for backward compatibility)
ITER_DECIDER_REGISTRY, register_iter_decider, get_iter_decider_class = create_registry("iter_decider", case_insensitive=True)

# Create updater registry
INPUT_UPDATER_REGISTRY, register_input_updater, get_input_updater_class = create_registry("input_updater", case_insensitive=True)

# Create loss func registry
LOSS_FUNC_REGISTRY, register_loss_func, get_loss_func_class = create_registry("loss_func", case_insensitive=True)

# Create output updater registry
OUTPUT_UPDATER_REGISTRY, register_output_updater, get_output_updater_class = create_registry("output_updater", case_insensitive=True)

# Create iter label generator registry
ITER_LABEL_GENERATOR_REGISTRY, register_iter_label_generator, get_iter_label_generator_class = create_registry("iter_label_generator", case_insensitive=True)


# Add specific type annotations for the get functions
if TYPE_CHECKING:
    def get_iter_decider_class(name: str) -> Type["IterDecider"]: ...
    def get_input_updater_class(name: str) -> Type["InputUpdater"]: ...
    def get_output_updater_class(name: str) -> Type["OutputUpdater"]: ...
    def get_loss_func_class(name: str) -> Type["LossFunc"]: ...
    def get_iter_label_generator_class(name: str) -> Type["IterLabelGenerator"]: ...


def capture_init_args(cls):
    """
    Decorator to capture initialization arguments of a model class.
    
    Args:
        cls: The class to decorate
        
    Returns:
        The decorated class with automatic init args capture
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Store all initialization arguments
        self._init_args = {}
        
        # Get parameter names from the original __init__ method
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._init_args[param_names[i]] = arg
        
        # Add keyword args
        self._init_args.update(kwargs)
        
        # Call the original __init__
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


def mark_wrapper_iter_decider(cls):
    """Decorator to mark an IterDecider as a wrapper over another decider.

    This flag allows builder logic (e.g., InterleavedIterDecider) to detect that the
    target class expects a base-decider instance or class and to perform special wiring.
    """
    try:
        setattr(cls, "_is_wrapper_iter_decider", True)
    except Exception:
        pass
    return cls