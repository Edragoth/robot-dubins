import numpy as np
from math import cos, sin, sqrt


class APF:
    """
    Campo de Potencial Artificial (Artificial Potential Field).
    Solo repulsión — sin objetivo de atracción.
    Calcula el giro óptimo para alejarse de obstáculos.
    """

    def __init__(self, speed=5.0, w_max=1.3, eta=1.0, d0=5.0):
        """
        Parámetros:
            speed:  velocidad fija del robot (m/s)
            w_max:  giro máximo permitido (rad/s)
            eta:    intensidad del campo repulsivo
            d0:     radio de influencia de cada obstáculo (m)
        """
        self.speed = speed
        self.w_max = w_max
        self.eta   = eta    # intensidad repulsiva
        self.d0    = d0     # radio de influencia

        # Obstáculos del playground — mismos que main.py
        self.obstaculos_rect = [
            {"x":  10.0, "y":  14.0, "w": 12.0, "h": 4.0},
            {"x":  -5.0, "y": -10.0, "w":  4.0, "h": 8.0},
            {"x":  10.0, "y": -10.0, "w":  8.0, "h": 4.0},
        ]
        self.obstaculos_circ = [
            {"x": -14.0, "y": 12.0, "r": 2.5},
        ]
        # Límite del playground como obstáculo implícito
        self.limite = 20.0

    def _distancia_punto_rectangulo(self, px, py, obs):
        """Distancia mínima de un punto al borde de un rectángulo."""
        x_min = obs["x"] - obs["w"] / 2
        x_max = obs["x"] + obs["w"] / 2
        y_min = obs["y"] - obs["h"] / 2
        y_max = obs["y"] + obs["h"] / 2

        # Punto más cercano dentro del rectángulo
        cx = max(x_min, min(px, x_max))
        cy = max(y_min, min(py, y_max))

        dx = px - cx
        dy = py - cy
        d  = sqrt(dx**2 + dy**2)

        # Gradiente apunta desde el obstáculo hacia el robot
        if d < 1e-6:
            return 0.0, 0.0, 0.0
        return d, dx / d, dy / d

    def _distancia_punto_circulo(self, px, py, obs):
        """Distancia mínima de un punto al borde de un círculo."""
        dx = px - obs["x"]
        dy = py - obs["y"]
        d_centro = sqrt(dx**2 + dy**2)
        d = max(d_centro - obs["r"], 0.0)

        if d_centro < 1e-6:
            return 0.0, 0.0, 0.0
        return d, dx / d_centro, dy / d_centro

    def _distancia_muro(self, px, py):
        """Distancia al muro perimetral y gradiente hacia el interior."""
        # Distancia a cada muro
        d_izq  = px - (-self.limite)   # muro izquierdo
        d_der  = self.limite - px       # muro derecho
        d_inf  = py - (-self.limite)    # muro inferior
        d_sup  = self.limite - py       # muro superior

        resultados = [
            (d_izq,  1.0,  0.0),   # gradiente apunta a la derecha
            (d_der, -1.0,  0.0),   # gradiente apunta a la izquierda
            (d_inf,  0.0,  1.0),   # gradiente apunta hacia arriba
            (d_sup,  0.0, -1.0),   # gradiente apunta hacia abajo
        ]
        return resultados

    def calcular_fuerza_repulsiva(self, px, py):
        """
        Calcula la fuerza repulsiva total en (px, py).
        Retorna (fx, fy) — vector de fuerza en coordenadas mundo.
        """
        fx_total = 0.0
        fy_total = 0.0

        # Repulsión de obstáculos rectangulares
        for obs in self.obstaculos_rect:
            d, gx, gy = self._distancia_punto_rectangulo(px, py, obs)
            if 0 < d < self.d0:
                mag = self.eta * (1.0/d - 1.0/self.d0) / (d**2)
                fx_total += mag * gx
                fy_total += mag * gy

        # Repulsión de obstáculos circulares
        for obs in self.obstaculos_circ:
            d, gx, gy = self._distancia_punto_circulo(px, py, obs)
            if 0 < d < self.d0:
                mag = self.eta * (1.0/d - 1.0/self.d0) / (d**2)
                fx_total += mag * gx
                fy_total += mag * gy

        # Repulsión del muro perimetral
        for d, gx, gy in self._distancia_muro(px, py):
            if 0 < d < self.d0:
                mag = self.eta * (1.0/d - 1.0/self.d0) / (d**2)
                fx_total += mag * gx
                fy_total += mag * gy

        return fx_total, fy_total

    def obtener_control_apf(self, x, y, theta_deg, w_usuario):
        """
        Calcula el control APF en la posición (x, y, theta).
        Combina la fuerza repulsiva con el giro del usuario.

        Retorna dict con V (distancia mínima), w resultante e info.
        """
        theta_rad = np.radians(theta_deg)

        # Fuerza repulsiva total
        fx, fy = self.calcular_fuerza_repulsiva(x, y)

        # Distancia mínima a cualquier obstáculo (como proxy de V)
        d_min = float('inf')
        for obs in self.obstaculos_rect:
            d, _, _ = self._distancia_punto_rectangulo(x, y, obs)
            d_min = min(d_min, d)
        for obs in self.obstaculos_circ:
            d, _, _ = self._distancia_punto_circulo(x, y, obs)
            d_min = min(d_min, d)
        for d, _, _ in self._distancia_muro(x, y):
            d_min = min(d_min, d)

        # Magnitud de la fuerza repulsiva
        f_mag = sqrt(fx**2 + fy**2)

        if f_mag < 1e-6:
            # Sin fuerza repulsiva — control total al usuario
            return {
                "V":          round(d_min, 4),
                "w":          round(float(w_usuario), 4),
                "w_usuario":  round(float(w_usuario), 4),
                "f_mag":      0.0,
                "peligroso":  False,
                "intervenido": False
            }

        # Proyectar fuerza repulsiva sobre el eje perpendicular al movimiento
        # El Dubins Car solo puede girar — proyectamos sobre la normal lateral
        # Normal lateral izquierda: (-sin(θ), cos(θ))
        f_lateral = -fx * sin(theta_rad) + fy * cos(theta_rad)

        # Convertir fuerza lateral en giro
        # Positivo → girar a la izquierda, Negativo → girar a la derecha
        w_repulsivo = float(np.clip(f_lateral, -self.w_max, self.w_max))

        # Combinar giro del usuario con repulsión
        # A mayor fuerza repulsiva, más domina el APF sobre el usuario
        peso_apf = min(1.0, f_mag / (self.eta * 2.0))
        w_final  = (1.0 - peso_apf) * w_usuario + peso_apf * w_repulsivo
        w_final  = float(np.clip(w_final, -self.w_max, self.w_max))

        return {
            "V":           round(d_min, 4),
            "w":           round(w_final, 4),
            "w_usuario":   round(float(w_usuario), 4),
            "w_repulsivo": round(w_repulsivo, 4),
            "f_mag":       round(f_mag, 4),
            "peligroso":   d_min < 2.0,
            "intervenido": abs(w_final - w_usuario) > 0.05
        }
