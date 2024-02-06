from typing import List

from particle_simulator.engine.event import event, eventclass


def test_subscribe_on_function() -> None:
    @event
    def broadcaster() -> str:
        return "Hello world!"

    calls: List[str] = []

    def callback(broadcast_message: str) -> None:
        calls.append(broadcast_message)

    broadcaster.subscribe(callback)
    assert broadcaster() == "Hello world!"
    assert calls == ["Hello world!"]


def test_subscribe_on_method() -> None:
    @eventclass
    class Class:
        @event
        def broadcaster(self) -> str:
            return "Hello world!"

    calls: List[str] = []

    def callback(broadcast_message: str) -> None:
        calls.append(broadcast_message)

    o = Class()
    o.broadcaster.subscribe(callback)
    o.broadcaster.unsubscribe(callback)
    assert o.broadcaster() == "Hello world!"
    assert calls == []


def test_subscribe_on_two_methods_does_not_interfere() -> None:
    @eventclass
    class Class:
        @event
        def broadcaster(self) -> str:
            return "Hello world!"

    calls: List[str] = []

    def callback(broadcast_message: str) -> None:
        calls.append(broadcast_message)

    a = Class()
    b = Class()
    a.broadcaster.subscribe(callback)
    b.broadcaster()
    assert calls == []
