from math import cos, sin, sqrt, hypot
import numpy as np


class CBFSDF:
    """
    CBF de alto orden (HOCBF) construido sobre la SDF del playground.
    No usa el BRT de HJ: todo se calcula analíticamente en tiempo real.

    Barrera:      h(x,y) = SDF(x,y) - margen        (h >= 0 es la zona segura)
    Dinámica:     x' = v cos(th),  y' = v sin(th),  th' = w

    Como h solo depende de (x,y), su primera derivada NO contiene a w
    (grado relativo 2), por eso se usa HOCBF:

        psi0 = h
        psi1 = h' + alpha1 * h
        condición: psi1' + alpha2 * psi1 >= 0

    Desarrollando, la condición queda lineal en w:  a*w + b >= 0
        a = v * (n . u_perp)
        b = v^2 * (u^T H u) + (alpha1 + alpha2) * h' + alpha1*alpha2*h
    con n = grad SDF, H = Hessiana de la SDF, u = (cos th, sin th),
    u_perp = (-sin th, cos th), h' = v * (n . u).

    El QP  min (w - w_usuario)^2  s.t.  a*w + b >= 0,  |w| <= w_max
    tiene solución analítica (1D).
    """

    def __init__(self, speed=5.0, w_max=1.3, limite=20.0):
        self.speed  = speed
        self.w_max  = w_max
        self.limite = limite

        # Mismos obstáculos que main.py
        self.obstaculos_rect = [
            {"x":  10.0, "y":  14.0, "w": 12.0, "h": 4.0},
            {"x":  -5.0, "y": -10.0, "w":  4.0, "h": 8.0},
            {"x":  10.0, "y": -10.0, "w":  8.0, "h": 4.0},
        ]
        self.obstaculos_circ = [
            {"x": -14.0, "y": 12.0, "r": 2.5},
        ]

    # ------------------------------------------------------------------
    # SDF de cada primitiva: devuelve (d, nx, ny, Hxx, Hxy, Hyy)
    # ------------------------------------------------------------------
    @staticmethod
    def _hess_esquina(nx, ny, dist):
        """Hessiana de una distancia a un punto: (I - n n^T) / dist."""
        return (1 - nx * nx) / dist, (-nx * ny) / dist, (1 - ny * ny) / dist

    def _sdf_rect(self, px, py, obs):
        dx, dy = px - obs["x"], py - obs["y"]
        qx = abs(dx) - obs["w"] / 2
        qy = abs(dy) - obs["h"] / 2
        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0

        if qx > 0 or qy > 0:                       # fuera del rectángulo
            ox, oy = max(qx, 0.0), max(qy, 0.0)
            d = hypot(ox, oy)
            if d < 1e-9:
                return 0.0, (sx if qx > qy else 0.0), (0.0 if qx > qy else sy), 0.0, 0.0, 0.0
            nx, ny = sx * ox / d, sy * oy / d
            if ox > 0 and oy > 0:                  # zona de esquina: curvatura 1/d
                hxx, hxy, hyy = self._hess_esquina(nx, ny, d)
            else:                                  # zona de cara plana: H = 0
                hxx = hxy = hyy = 0.0
            return d, nx, ny, hxx, hxy, hyy

        # dentro del rectángulo
        d = max(qx, qy)
        if qx > qy:
            return d, sx, 0.0, 0.0, 0.0, 0.0
        return d, 0.0, sy, 0.0, 0.0, 0.0

    def _sdf_circ(self, px, py, obs):
        dx, dy = px - obs["x"], py - obs["y"]
        dc = hypot(dx, dy)
        if dc < 1e-9:
            return -obs["r"], 1.0, 0.0, 0.0, 0.0, 0.0
        nx, ny = dx / dc, dy / dc
        hxx, hxy, hyy = self._hess_esquina(nx, ny, dc)
        return dc - obs["r"], nx, ny, hxx, hxy, hyy

    def _sdf_muro(self, px, py):
        """Distancia al muro del cuadrado [-L, L]^2 (positiva dentro)."""
        d = self.limite - max(abs(px), abs(py))
        if abs(px) >= abs(py):
            return d, (-1.0 if px >= 0 else 1.0), 0.0, 0.0, 0.0, 0.0
        return d, 0.0, (-1.0 if py >= 0 else 1.0), 0.0, 0.0, 0.0

    def sdf(self, px, py):
        """SDF de la unión de peligros = mínimo. Retorna (d, nx, ny, Hxx, Hxy, Hyy)."""
        mejor = self._sdf_muro(px, py)
        for obs in self.obstaculos_rect:
            r = self._sdf_rect(px, py, obs)
            if r[0] < mejor[0]:
                mejor = r
        for obs in self.obstaculos_circ:
            r = self._sdf_circ(px, py, obs)
            if r[0] < mejor[0]:
                mejor = r
        return mejor

    # ------------------------------------------------------------------
    # Control HOCBF
    # ------------------------------------------------------------------
    def obtener_control(self, x, y, theta_rad, w_usuario,
                        alpha1=1.0, alpha2=1.0, margen=0.0):
        d, nx, ny, hxx, hxy, hyy = self.sdf(x, y)
        h = d - margen

        c, s = cos(theta_rad), sin(theta_rad)
        v = self.speed

        n_u    = nx * c + ny * s            # n . u
        n_uper = -nx * s + ny * c           # n . u_perp
        h_dot  = v * n_u
        psi1   = h_dot + alpha1 * h
        uHu    = hxx * c * c + 2 * hxy * c * s + hyy * s * s

        a = v * n_uper
        b = v * v * uHu + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h

        w_usr = float(np.clip(w_usuario, -self.w_max, self.w_max))

        # QP 1D: min (w - w_usr)^2  s.t.  a*w + b >= 0
        if abs(a) < 1e-9:
            w = w_usr
        elif a > 0:
            w = max(w_usr, -b / a)
        else:
            w = min(w_usr, -b / a)
        w = float(np.clip(w, -self.w_max, self.w_max))

        factible = (a * w + b) >= -1e-9
        if not factible:
            # La restricción no se puede cumplir con |w| <= w_max:
            # mejor esfuerzo = maximizar a*w + b (giro máximo hacia el lado útil)
            w = self.w_max * (1.0 if a >= 0 else -1.0)

        return {
            "h":           round(h, 4),
            "psi1":        round(psi1, 4),
            "V":           round(psi1, 4),       # el main.py lo usa como "valor" a mostrar
            "w_usuario":   round(w_usr, 4),
            "w":           round(w, 4),
            "factible":    bool(factible),
            "peligroso":   psi1 < 0,
            "intervenido": abs(w - w_usr) > 1e-4,
        }
