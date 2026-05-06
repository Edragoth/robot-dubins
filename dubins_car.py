import numpy as np
from math import cos, sin, pi

class DubinsCar:
    """Modelo cinemático de un Dubins Car."""
#
    def __init__(self, x=0.0, y=0.0, theta=0.0, v_max=2.0, w_max=1.5):
        self.x     = x
        self.y     = y
        self.theta = theta
        self.v     = 0.0
        self.w     = 0.0
        self.v_max = v_max
        self.w_max = w_max

    def actualizar(self, v, w, dt):
        

        self.x     += v * cos(self.theta) * dt
        self.y     += v * sin(self.theta) * dt
        self.theta += w * dt

        self.v = v
        self.w = w

    def reset(self, x=0.0, y=0.0, theta=0.0):
        self.x     = x
        self.y     = y
        self.theta = theta
        self.v     = 0.0
        self.w     = 0.0

    def obtener_estado(self):
        return {
            "x":     round(self.x, 3),
            "y":     round(self.y, 3),
            "theta": round(self.theta, 3),
            "v":     round(self.v, 3),
            "w":     round(self.w, 3),
        }