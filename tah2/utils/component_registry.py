import functools
import inspect
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, TypeVar

if TYPE_CHECKING:
    from tah2.model.input_updater import InputUpdater
    from tah2.model.iter_decider import IterDecider
    from tah2.model.iter_label import IterLabelGenerator
    from tah2.model.loss import LossFunc

T = TypeVar("T")


def create_registry(
    registry_name: str,
    case_insensitive: bool = False,
) -> Tuple[Dict[str, Type[T]], Callable, Callable[[str], Type[T]]]:
    registry: Dict[str, Type[T]] = {}

    def register(cls_or_name=None, name: Optional[str] = None):
        def _register(c: Type[T]) -> Type[T]:
            if isinstance(cls_or_name, str):
                class_name = cls_or_name
            elif name is not None:
                class_name = name
            else:
                class_name = c.__name__

            registry[class_name] = c
            if case_insensitive:
                registry[class_name.lower()] = c
            return c

        if cls_or_name is not None and not isinstance(cls_or_name, str):
            return _register(cls_or_name)
        return _register

    def get_class(name: str) -> Type[T]:
        key = name if name in registry else (name.lower() if case_insensitive else name)
        if key not in registry:
            raise ValueError(
                f"Unknown {registry_name} class: {name}. Available: {list(registry.keys())}"
            )
        return registry[key]

    return registry, register, get_class


ITER_DECIDER_REGISTRY, register_iter_decider, get_iter_decider_class = create_registry(
    "iter_decider",
    case_insensitive=True,
)
INPUT_UPDATER_REGISTRY, register_input_updater, get_input_updater_class = (
    create_registry(
        "input_updater",
        case_insensitive=True,
    )
)
LOSS_FUNC_REGISTRY, register_loss_func, get_loss_func_class = create_registry(
    "loss_func",
    case_insensitive=True,
)
(
    ITER_LABEL_GENERATOR_REGISTRY,
    register_iter_label_generator,
    get_iter_label_generator_class,
) = create_registry(
    "iter_label_generator",
    case_insensitive=True,
)


if TYPE_CHECKING:

    def get_iter_decider_class(name: str) -> Type["IterDecider"]: ...
    def get_input_updater_class(name: str) -> Type["InputUpdater"]: ...
    def get_loss_func_class(name: str) -> Type["LossFunc"]: ...
    def get_iter_label_generator_class(name: str) -> Type["IterLabelGenerator"]: ...


def capture_init_args(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        self._init_args = {}
        param_names = list(inspect.signature(original_init).parameters.keys())[1:]
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._init_args[param_names[i]] = arg
        self._init_args.update(kwargs)
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


def mark_wrapper_iter_decider(cls):
    try:
        setattr(cls, "_is_wrapper_iter_decider", True)
    except Exception:
        pass
    return cls
