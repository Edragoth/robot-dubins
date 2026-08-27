import numpy as np
from math import cos, sin, sqrt, atan2


class APF:
    """
    Campo de Potencial Artificial (Artificial Potential Field).
    Solo repulsión — sin objetivo de atracción.
    """

    def __init__(self, speed=5.0, w_max=1.3, eta=1.0, d0=8.0):
        self.speed = speed
        self.w_max = w_max
        self.eta   = eta
        self.d0    = d0

        # Parámetros especiales para el muro perimetral
        self.ETA_MURO = 3.0
        self.D0_MURO  = 10.0

        self.obstaculos_rect = [
            {"x":  10.0, "y":  14.0, "w": 12.0, "h": 4.0},
            {"x":  -5.0, "y": -10.0, "w":  4.0, "h": 8.0},
            {"x":  10.0, "y": -10.0, "w":  8.0, "h": 4.0},
        ]
        self.obstaculos_circ = [
            {"x": -14.0, "y": 12.0, "r": 2.5},
        ]
        self.limite = 20.0

        self.D_CONTROL_TOTAL = 4.0
        self.D_MIN           = 0.3
        self.F_MAX           = 10.0
        self.F_REF           = 1.0

    def _distancia_punto_rectangulo(self, px, py, obs):
        x_min = obs["x"] - obs["w"] / 2
        x_max = obs["x"] + obs["w"] / 2
        y_min = obs["y"] - obs["h"] / 2
        y_max = obs["y"] + obs["h"] / 2
        cx = max(x_min, min(px, x_max))
        cy = max(y_min, min(py, y_max))
        dx = px - cx
        dy = py - cy
        d  = sqrt(dx**2 + dy**2)
        if d < 1e-6:
            return 0.0, 0.0, 0.0
        return d, dx / d, dy / d

    def _distancia_punto_circulo(self, px, py, obs):
        dx = px - obs["x"]
        dy = py - obs["y"]
        d_centro = sqrt(dx**2 + dy**2)
        d = max(d_centro - obs["r"], 0.0)
        if d_centro < 1e-6:
            return 0.0, 0.0, 0.0
        return d, dx / d_centro, dy / d_centro

    def _distancia_muro(self, px, py):
        return [
            (px - (-self.limite),  1.0,  0.0),
            (self.limite - px,    -1.0,  0.0),
            (py - (-self.limite),  0.0,  1.0),
            (self.limite - py,     0.0, -1.0),
        ]

    def calcular_fuerza_repulsiva(self, px, py):
        fx_total = 0.0
        fy_total = 0.0

        # Obstáculos rectangulares
        for obs in self.obstaculos_rect:
            d, gx, gy = self._distancia_punto_rectangulo(px, py, obs)
            d = max(d, self.D_MIN)
            if d < self.d0:
                mag = self.eta * (1.0/d - 1.0/self.d0) / (d**2)
                fx_total += mag * gx
                fy_total += mag * gy

        # Obstáculos circulares
        for obs in self.obstaculos_circ:
            d, gx, gy = self._distancia_punto_circulo(px, py, obs)
            d = max(d, self.D_MIN)
            if d < self.d0:
                mag = self.eta * (1.0/d - 1.0/self.d0) / (d**2)
                fx_total += mag * gx
                fy_total += mag * gy

        # Muro perimetral — parámetros más agresivos
        for d, gx, gy in self._distancia_muro(px, py):
            d = max(d, self.D_MIN)
            if d < self.D0_MURO:
                mag = self.ETA_MURO * (1.0/d - 1.0/self.D0_MURO) / (d**2)
                fx_total += mag * gx
                fy_total += mag * gy

        # Limitar magnitud máxima
        f_mag = sqrt(fx_total**2 + fy_total**2)
        if f_mag > self.F_MAX:
            fx_total = fx_total / f_mag * self.F_MAX
            fy_total = fy_total / f_mag * self.F_MAX
            f_mag    = self.F_MAX

        return fx_total, fy_total, f_mag

    def obtener_control_apf(self, x, y, theta_deg, w_usuario):
        theta_rad = np.radians(theta_deg)

        fx, fy, f_mag = self.calcular_fuerza_repulsiva(x, y)

        # Distancia mínima a cualquier obstáculo o muro
        d_min = float('inf')
        for obs in self.obstaculos_rect:
            d, _, _ = self._distancia_punto_rectangulo(x, y, obs)
            d_min = min(d_min, d)
        for obs in self.obstaculos_circ:
            d, _, _ = self._distancia_punto_circulo(x, y, obs)
            d_min = min(d_min, d)
        for d, _, _ in self._distancia_muro(x, y):
            d_min = min(d_min, d)

        if f_mag < 1e-6:
            return {
                "V":           round(d_min, 4),
                "w":           round(float(w_usuario), 4),
                "w_usuario":   round(float(w_usuario), 4),
                "f_mag":       0.0,
                "peligroso":   False,
                "intervenido": False
            }

        # Ángulo de escape = dirección de la fuerza repulsiva
        theta_escape = atan2(fy, fx)

        # Diferencia angular normalizada a [-pi, pi]
        delta_theta = theta_escape - theta_rad
        while delta_theta >  np.pi: delta_theta -= 2*np.pi
        while delta_theta < -np.pi: delta_theta += 2*np.pi

        # Signo determina dirección de giro
        w_repulsivo = float(np.sign(delta_theta) * self.w_max)

        # Control total si está muy cerca, mezcla gradual si está lejos
        if d_min < self.D_CONTROL_TOTAL:
            w_final = w_repulsivo
        else:
            peso_apf = min(1.0, f_mag / self.F_REF)
            w_final  = (1.0 - peso_apf) * w_usuario + peso_apf * w_repulsivo

        w_final = float(np.clip(w_final, -self.w_max, self.w_max))

        return {
            "V":           round(d_min, 4),
            "w":           round(w_final, 4),
            "w_usuario":   round(float(w_usuario), 4),
            "w_repulsivo": round(w_repulsivo, 4),
            "f_mag":       round(f_mag, 4),
            "peligroso":   d_min < 2.0,
            "intervenido": abs(w_final - w_usuario) > 0.05
        }
