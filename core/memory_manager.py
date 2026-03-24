from typing import List

from .message import Message


class MemoryManager:
    def __init__(self) -> None:
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def all(self) -> List[Message]:
        return list(self._messages)
