"""A module that defines decorators to transform functions and methods
into events"""

from types import MethodType
from typing import Callable, TypeVar, Generic, Any, Type, Set, List

_T = TypeVar("_T")


class _Event(Generic[_T]):
    def __init__(self, broadcaster: Callable[..., _T]) -> None:
        self._callbacks: Set[Callable[[_T], None]] = set()
        self.broadcaster = broadcaster

    def subscribe(self, callback: Callable[[_T], None]) -> None:
        """Register the given callable as a subscriber.

        :param callback: a function-like object that will be called
            each time the event is triggered. It must be hashable
        """
        self._callbacks.add(callback)

    def unsubscribe(self, callback: Callable[[_T], None]) -> None:
        """Unregister the given callable.

        :param callback: the subscriber to unregister
        :raises KeyError: if the given callable in not registered
        """
        self._callbacks.remove(callback)

    def __call__(self, *args: Any, **kwargs: Any) -> _T:
        try:
            result = self.broadcaster(*args, **kwargs)
        except TypeError as exception:
            raise TypeError(
                f"{exception}\nIf this is a method and `self` is missing, "
                f"consider adding the `eventclass` decorator on top of the "
                f"declaring class."
            ) from exception
        for callback in self._callbacks:
            callback(result)
        return result


def _duplicate_events(obj: object) -> None:
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if not isinstance(attr, _Event):
            continue
        # Creating a new Event specifically for "obj" to
        # avoid side effects on other instances:
        event_for_instance = _Event(attr.broadcaster)
        setattr(
            obj,
            attr_name,
            MethodType(event_for_instance, obj),
        )


def event(broadcaster: Callable[..., _T]) -> _Event[_T]:
    """A decorator that transforms a function into an event.
    Other functions ("subscribers") can subscribe to this event,
    and will be called each time this event is triggered. The
    subscribers must take a single argument that corresponds
    to the output of the event.

    .. warning::
        When decorating a method, the declaring class should be decorated
        with an :func:`eventclass`

    >>> calls: List[str] = []
    >>>
    >>> @event
    ... def on_greetings() -> str:
    ...     return "Hello"
    >>>
    >>> def append_target(broadcaster_output: str) -> None:
    ...     calls.append(f"{broadcaster_output} world!")
    >>>
    >>> on_greetings.subscribe(append_target)
    >>> on_greetings()
    'Hello'
    >>> calls
    ['Hello world!']

    :param broadcaster: The function to decorate, to turn it into an event
    :return: An event, which other functions can subscribe to or unsubscribe from
    """
    return _Event(broadcaster)


def eventclass(class_: Type[_T]) -> Type[_T]:
    """A class decorator to enable the decoration of methods with :func:`event`

    >>> calls: List[str] = []
    >>>
    ... @eventclass
    ... class GUI:
    ...     @event
    ...     def on_button_click(self) -> str:
    ...         return "Clicked"
    >>>
    >>> def append_target(broadcaster_output: str) -> None:
    ...     calls.append(f"{broadcaster_output} to say hello!")
    >>>
    >>> gui = GUI()
    >>> gui.on_button_click.subscribe(append_target)
    >>> gui.on_button_click()
    'Clicked'
    >>> calls
    ['Clicked to say hello!']

    :param class_: The class to decorate
    :return: The corresponding subclass that supports event methods
    """

    class EventClass(class_):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            _duplicate_events(self)

    return EventClass
