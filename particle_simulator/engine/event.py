from types import MethodType
from typing import Callable, List, TypeVar, Generic, Any, Type

_T = TypeVar("_T")


class _Event(Generic[_T]):
    def __init__(self, broadcaster: Callable[..., _T]) -> None:
        self._callbacks: List[Callable[[_T], None]] = []
        self.broadcaster = broadcaster

    def subscribe(self, callback: Callable[[_T], None]) -> None:
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[_T], None]) -> None:
        self._callbacks.remove(callback)

    def __call__(self, *args: Any, **kwargs: Any) -> _T:
        result = self.broadcaster(*args, **kwargs)
        for callback in self._callbacks:
            callback(result)
        return result


def _duplicate_events(obj: object) -> None:
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if not isinstance(attr, _Event):
            continue
        # Creating a new _Event specifically for "obj" to
        # avoid side effects on other instances:
        event_for_instance = _Event(attr.broadcaster)
        setattr(
            obj,
            attr_name,
            MethodType(event_for_instance, obj),
        )


def event(broadcaster: Callable[..., _T]) -> _Event[_T]:
    return _Event(broadcaster)


def eventclass(class_: Type[_T]) -> Type[_T]:
    class EventClass(class_):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            _duplicate_events(self)

    return EventClass
