from enum import Enum

class SENSORS(Enum):
    """
    è una lista dei sensori presenti nel dataset
    serve solo per evitare di scrivere a mano i nomi dei sensori
    """

    Altitude = "Altitude"
    Mach = "Mach"
    Pamb = "Pamb"
    Pt2 = "Pt2"
    TAT = "TAT"
    WFuel = "WFuel"
    VAFN = "VAFN"
    VBV = "VBV"
    Fan_Speed = "Fan_Speed"
    Core_Speed = "Core_Speed"
    T25 = "T25"
    T3 = "T3"
    Ps3 = "Ps3"
    T45 = "T45"
    P25 = "P25"
    T5 = "T5"

    @classmethod
    def values(cls) -> list[str]:
        """Ritorna la lista dei valori."""
        return [e.value for e in cls]

    @classmethod
    def iter(cls) -> list[str]:
        """DEPRECATO, usa values. Ritorna la lista dei valori."""
        return [e.value for e in cls]

    @classmethod
    def members(cls) -> list["ESENSORS"]:
        """Ritorna la lista dei membri Enum."""
        return list(cls)

class ESENSORS(Enum):
    """
    è una lista dei sensori presenti nel dataset
    serve solo per evitare di scrivere a mano i nomi dei sensori
    """

    Sensed_Altitude = "Sensed_Altitude"
    Sensed_Mach = "Sensed_Mach"
    Sensed_Pamb = "Sensed_Pamb"
    Sensed_Pt2 = "Sensed_Pt2"
    Sensed_TAT = "Sensed_TAT"
    Sensed_WFuel = "Sensed_WFuel"
    Sensed_VAFN = "Sensed_VAFN"
    Sensed_VBV = "Sensed_VBV"
    Sensed_Fan_Speed = "Sensed_Fan_Speed"
    Sensed_Core_Speed = "Sensed_Core_Speed"
    Sensed_T25 = "Sensed_T25"
    Sensed_T3 = "Sensed_T3"
    Sensed_Ps3 = "Sensed_Ps3"
    Sensed_T45 = "Sensed_T45"
    Sensed_P25 = "Sensed_P25"
    Sensed_T5 = "Sensed_T5"

    @classmethod
    def values(cls) -> list[str]:
        """Ritorna la lista dei valori."""
        return [e.value for e in cls]

    @classmethod
    def iter(cls) -> list[str]:
        """DEPRECATO, usa values. Ritorna la lista dei valori."""
        return [e.value for e in cls]

    @classmethod
    def members(cls) -> list["ESENSORS"]:
        """Ritorna la lista dei membri Enum."""
        return list(cls)



_REPAIR_EVENT_TYPES_STR = ["ww", "hpt", "hpc"]
class RepairEventType(Enum):
    WW = 0
    HPT = 1
    HPC = 2

    @classmethod
    def values(cls) -> list[str]:
        """Ritorna la lista dei valori."""
        return [str(e.value) for e in cls]

    def __str__(self) -> str:
        return _REPAIR_EVENT_TYPES_STR[self.value]

class RepairEventType(Enum):
    WW = 0
    HPT = 1
    HPC = 2

    @classmethod
    def values(cls) -> list[str]:
        """Ritorna la lista dei valori."""
        return [str(e.value) for e in cls]

    def __str__(self) -> str:
        return _REPAIR_EVENT_TYPES_STR[self.value]


_SNAPSHOTS_LONG_STR = [
"Standing",
"Pushback/Towing",
"Taxi",
"Takeoff",
"Initial Climb", 
"En Route",
"Approach",
"Landing"
]

_SNAPSHOTS_SHORT_STR = [
"STD",
"PBT",
"TXI",
"TOF",
"ICL", 
"ENR",
"APR",
"LDG"
]



class Snapshots(Enum):
    STD = 1 # Standing
    PBT = 2 # Pushback/Towing
    TXI = 3 # Taxi
    TOF = 4 # Takeoff
    ICL = 5 # Initial Climb
    ENR = 6 # En Route
    APR = 7 # Approach
    LDG = 8 # Landing

    @classmethod
    def values(cls) -> list[int]:
        """Ritorna la lista dei valori."""
        return [e.value for e in cls]

    def to_def(self):
        return _SNAPSHOTS_LONG_STR[self.value]

    def __str__(self) -> str:
        return _SNAPSHOTS_SHORT_STR[self.value]
