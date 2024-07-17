from dataclasses import dataclass

@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

@dataclass
class Vector3:
    x: float
    y: float
    z: float

@dataclass
class IMU:
    sensor_id: int
    orientation: Quaternion
    acceleration: Vector3
    angular_velocity: Vector3
    magnetic_field: Vector3
    


