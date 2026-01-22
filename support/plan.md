# FSU Spring 2026 Capstone Plan

- [FSU Spring 2026 Capstone Plan](#fsu-spring-2026-capstone-plan)
  - [0. Considerations](#0-considerations)
    - [Power](#power)
      - [Supply](#supply)
    - [Heat](#heat)
  - [1. Simulation Environment (Python)](#1-simulation-environment-python)
    - [Device State](#device-state)
    - [Processor](#processor)


## 0. Considerations
### Power
- **Power budget**: Pi 5 (~5W) + TPU (~2W) + GPS (~0.1W) + Camera (~1W) + LEDs (~0.5W) ≈ 8-10W peak

#### Supply
- Direct connection (fusebox/cigarette lighter):
  - 12V-to-5V converter

- Indirect connection:
  - Limit capabilities (capture only)

### Heat
- **Passive cooling**: Heatsink on Pi 5 SoC
- **Active cooling**: 5V fan triggered at 60°C
- **Thermal throttling**: Reduce inference rate at 70°C
- **Emergency mode**: At 80°C, stop ML processing, continue recording only
- **Critical Stop**: At 85°C, safe time "sleep"
- Maybe thermal pad between Pi and metal enclosure

## 1. Simulation Environment (Python)

### Device State

**Create a class to store device state**
```python
from enum import IntFlag, auto

class DeviceStatus(IntFlag):
    """Bitfield for device status flags"""
    NONE = 0
    RECORDING = auto()          # 0x01
    GPS_LOCKED = auto()         # 0x02
    TPU_READY = auto()          # 0x04
    OVERHEATING = auto()        # 0x08
    LOW_STORAGE = auto()        # 0x10
    ERROR = auto()              # 0x20

class BlindSpotStatus(IntFlag):
    """Blind spot detection states"""
    CLEAR = 0
    DRIVER_WARNING = auto()     # Object in driver blind spot
    PASSENGER_WARNING = auto()  # Object in passenger blind spot

class CollisionStatus(IntFlag):
    """Forward collision states"""
    SAFE = 0                    # Green - no threat
    WARNING = auto()            # Yellow - approaching object
    EMERGENCY = auto()          # Red - immediate threat

class ConnectionStatus(IntFlag):
    """Device connection states"""
    DISCONNECTED = 0
    CONNECTING = auto()
    CONNECTED = auto()

@dataclass
class GPSData:
    """GPS information container"""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed_mph: float = 0.0
    heading: float = 0.0
    satellites: int = 0
    fix_quality: int = 0
    timestamp: Optional[datetime] = None

@dataclass
class DetectionResult:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    zone: str  # "left", "center", "right"
    distance_estimate: Optional[float] = None  # meters

@dataclass
class DashCamState:
    # Status flags
    status: DeviceStatus = DeviceStatus.NONE
    blind_spot: BlindSpotStatus = BlindSpotStatus.CLEAR
    collision: CollisionStatus = CollisionStatus.SAFE
    connection: ConnectionStatus = ConnectionStatus.DISCONNECTED
    
    # Sensor data
    temperature_c: float = 0.0
    gps: GPSData = field(default_factory=GPSData)
    
    # Recording info
    storage_used_percent: float = 0.0
```

### Processor