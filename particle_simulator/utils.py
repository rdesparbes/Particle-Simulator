from typing import Callable, Any


def any_args(action: Callable[[], None]) -> Callable[..., None]:
    def wrapper(*_args: Any, **_kwargs: Any) -> None:
        return action()

    return wrapper
