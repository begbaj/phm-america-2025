class PlotData:
    esn: int
    snap: int
    sensor: str
    size: tuple[float,float] = (15,12)
    cols: int = 3
    repair: str


    def __init__(self, esn=0, snap=0, sensor="None") -> None:
        self.esn = esn
        self.snap = snap
        self.sensor = sensor