from enum import Enum

BLUE = "#0D7FF2"
GREEN = "#03FC4E"
YELLOW = "#FFF600"
RED = "#FF0018"


class Status(Enum):
    CONNECTING = 0
    CONNECTED = 1
    ERROR = 2

    SAFE = 3
    WARNING = 4
    EMERGENCY = 5


LED_COLOR_MAP = {
    Status.CONNECTING: BLUE,
    Status.CONNECTED: GREEN,
    Status.ERROR: RED,
    Status.SAFE: GREEN,
    Status.WARNING: YELLOW,
    Status.EMERGENCY: RED,
}


class DashCam:
    def __init__(self) -> None:
        self.connect()

    def connect(self, timeout: int | None = None) -> None:
        self.status = Status.CONNECTED

    def color(self, color_override: Status | None = None) -> str:
        if color_override:
            return LED_COLOR_MAP[color_override]

        return LED_COLOR_MAP[self.status]

    def process(self) -> None:
        pass


def start():
    dashcam = DashCam()

    retries = 5
    attempts = 0
    while not dashcam.status == Status.CONNECTED and attempts < retries:
        dashcam.connect()

    while dashcam.status == Status.CONNECTED:
        dashcam.process()


if __name__ == "__main__":
    start()
