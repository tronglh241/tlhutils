from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Item:
    def __init__(
        self,
        timestamp: float,
        value: Any,
        new: bool = True,
    ):
        self.timestamp = timestamp
        self.value = value
        self.new = new


class Source(ABC):
    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def next(self) -> Optional[Item]:
        pass


class Synchronizer:
    def __init__(
        self,
        sources: Dict[str, Source],
    ):
        self.sources = sources

    def __iter__(self) -> Synchronizer:
        for source in self.sources.values():
            source.reset()

        self.last_items: Dict[str, Item] = {name: Item(-1.0, None) for name in self.sources}
        self.incoming_items: Dict[str, Optional[Item]] = {name: source.next() for name, source in self.sources.items()}

        return self

    def __next__(self) -> Dict[str, Item]:
        if any(item is None for item in self.incoming_items.values()):
            raise StopIteration

        ref_name = min(self.incoming_items.items(), key=lambda item: item[1].timestamp)[0]  # type: ignore
        ref_item = self.incoming_items[ref_name]
        assert ref_item is not None
        ref_item.new = True

        result: Dict[str, Item] = {ref_name: ref_item}

        for name, incoming_item in self.incoming_items.items():
            if name == ref_name:
                continue

            last_item = self.last_items[name]

            assert ref_item is not None
            assert incoming_item is not None
            assert last_item is not None

            if abs(ref_item.timestamp - incoming_item.timestamp) < abs(ref_item.timestamp - last_item.timestamp):
                result[name] = incoming_item
            else:
                result[name] = last_item

            if abs(ref_item.timestamp - result[name].timestamp) > 1e-6:
                result[name].new = False

        for name, item in result.items():
            if item.new:
                self.last_items[name] = self.incoming_items[name]  # type: ignore
                self.incoming_items[name] = self.sources[name].next()

        return result
