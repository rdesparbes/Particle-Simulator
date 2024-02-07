from typing import Type
from unittest.mock import MagicMock

import pytest

from particle_simulator.engine.event import event, eventclass, _Event


@pytest.fixture(name="on_say_hello")
def fixture_on_say_hello() -> _Event[str]:
    @event
    def on_say_hello() -> str:
        return "Hello world!"

    return on_say_hello


@pytest.fixture(name="event_class")
def fixture_event_class() -> Type:
    @eventclass
    class Class:
        @event
        def on_say_hello_to(self, name: str) -> str:
            return f"Hello {name}!"

    return Class


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
    class Class: ...

    event_obj = Class()

    assert isinstance(event_obj, Class)


def test_subscribe_given_method(event_class: Type) -> None:
    callback = MagicMock()
    event_obj = event_class()

    event_obj.on_say_hello_to.subscribe(callback)

    assert event_obj.on_say_hello_to("world") == "Hello world!"
    callback.assert_called_once_with("Hello world!")


def test_subscribe_given_two_instances_keeps_events_isolated(event_class: Type) -> None:
    a = event_class()
    b = event_class()
    callback_a = MagicMock()
    callback_b = MagicMock()

    a.on_say_hello_to.subscribe(callback_a)
    b.on_say_hello_to.subscribe(callback_b)

    assert a.on_say_hello_to(name="'a'") == "Hello 'a'!"
    callback_a.assert_called_once_with("Hello 'a'!")
    callback_b.assert_not_called()


def test_forgetting_event_class_raises_type_error(event_class: Type) -> None:
    event_obj = event_class()
    with pytest.raises(TypeError, match="eventclass"):
        event_obj.on_say_hello_to()
