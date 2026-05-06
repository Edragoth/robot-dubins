from math import cos, sin, atan2, sqrt, pi
from dubins_car import DubinsCar

class DubinsPlanner:
    """Calcula la trayectoria más corta entre dos puntos para un Dubins Car."""

    def __init__(self, radio_min=0.5):
        self.radio_min   = radio_min
        self.trayectoria = []

    def calcular_angulo(self, x1, y1, x2, y2):
        """Calcula el ángulo entre dos puntos."""
        return atan2(y2 - y1, x2 - x1)

    def calcular_distancia(self, x1, y1, x2, y2):
        """Calcula la distancia entre dos puntos."""
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def planificar(self, x_ini, y_ini, theta_ini, x_fin, y_fin, theta_fin, pasos=200):
        """Genera la trayectoria completa desde el punto inicial al final."""
        self.trayectoria = []

        robot = DubinsCar(x=x_ini, y=y_ini, theta=theta_ini)

        angulo_objetivo  = self.calcular_angulo(x_ini, y_ini, x_fin, y_fin)
        distancia_total  = self.calcular_distancia(x_ini, y_ini, x_fin, y_fin)

        for i in range(pasos):
            estado = robot.obtener_estado()
            self.trayectoria.append((estado["x"], estado["y"]))

            dx     = x_fin - estado["x"]
            dy     = y_fin - estado["y"]
            angulo = atan2(dy, dx)

            diff_angulo = angulo - estado["theta"]
            while diff_angulo >  pi: diff_angulo -= 2 * pi
            while diff_angulo < -pi: diff_angulo += 2 * pi

            w = max(-robot.w_max, min(robot.w_max, diff_angulo * 2.0))
            v = robot.v_max * (1.0 - abs(diff_angulo) / pi)

            robot.actualizar(v, w, dt=0.05)

        return self.trayectoria