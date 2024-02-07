from typing import Type, Protocol
from unittest.mock import MagicMock

import pytest

from particle_simulator.engine.event import event, eventclass, _Event


@pytest.fixture(name="on_say_hello")
def fixture_on_say_hello() -> _Event[str]:
    @event
    def on_say_hello() -> str:
        return "Hello world!"

    return on_say_hello


@eventclass
class Greeter(Protocol):
    @event
    def on_say_hello_to(self, name: str) -> str: ...


@pytest.fixture(name="greeter")
def fixture_greeter() -> Type[Greeter]:
    @eventclass
    class _Greeter:
        @event
        def on_say_hello_to(self, name: str) -> str:
            return f"Hello {name}!"

    return _Greeter


def test_subscribe_given_function(on_say_hello: _Event[str]) -> None:
    callback = MagicMock()

    on_say_hello.subscribe(callback)

    assert on_say_hello() == "Hello world!"
    callback.assert_called_once_with("Hello world!")


def test_unsubscribe_given_function(on_say_hello: _Event[str]) -> None:
    callback = MagicMock()

    on_say_hello.subscribe(callback)
    on_say_hello.unsubscribe(callback)

    assert on_say_hello() == "Hello world!"
    callback.assert_not_called()


def test_unsubscribe_given_unregistered_function_raises_key_error(
    on_say_hello: _Event[str],
) -> None:
    callback = MagicMock()

    with pytest.raises(KeyError):
        on_say_hello.unsubscribe(callback)


def test_instance_of_decorated_class_does_not_change_of_type() -> None:
    @eventclass
    class EventClass: ...

    event_obj = EventClass()

    assert isinstance(event_obj, EventClass)


def test_subscribe_given_method(greeter: Type[Greeter]) -> None:
    callback = MagicMock()
    event_obj = greeter()

    event_obj.on_say_hello_to.subscribe(callback)

    assert event_obj.on_say_hello_to("world") == "Hello world!"
    callback.assert_called_once_with("Hello world!")


def test_subscribe_given_two_instances_keeps_events_isolated(
    greeter: Type[Greeter],
) -> None:
    a = greeter()
    b = greeter()
    callback_a = MagicMock()
    callback_b = MagicMock()

    a.on_say_hello_to.subscribe(callback_a)
    b.on_say_hello_to.subscribe(callback_b)

    assert a.on_say_hello_to(name="'a'") == "Hello 'a'!"
    callback_a.assert_called_once_with("Hello 'a'!")
    callback_b.assert_not_called()


def test_forgetting_event_class_raises_type_error(
    greeter: Type[Greeter],
) -> None:
    event_obj = greeter()
    with pytest.raises(TypeError, match="eventclass"):
        event_obj.on_say_hello_to()
