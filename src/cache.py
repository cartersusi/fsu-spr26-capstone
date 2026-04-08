from collections import deque


class Cache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.items = deque(maxlen=max_size)

    def set_max_size(self, max_size: int) -> None:
        self.max_size = max_size
        self.items = deque(self.items, maxlen=max_size)

    def add(self, item) -> None:
        self.items.append(item)

    def get_all(self):
        return list(self.items)

    def clear(self) -> None:
        self.items.clear()

    def __len__(self):
        return len(self.items)
