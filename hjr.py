import numpy as np
import h5py


class DubinsHJR:
    """
    Carga el BRT pre-calculado desde MATLAB (brt_result.mat)
    y lo sirve para visualización y control de seguridad (LRF, CBF).
    """

    def __init__(self, speed=5.0, w_max=1.3):
        self.speed    = speed
        self.w_max    = w_max
        self.data     = None   # shape: (T, Nx, Ny, Ntheta)
        self.dVdx     = None   # shape: (Nx, Ny, Ntheta)
        self.dVdy     = None   # shape: (Nx, Ny, Ntheta)
        self.dVdtheta = None   # shape: (Nx, Ny, Ntheta)
        self.xs       = None
        self.ys       = None
        self.thetas   = None
        self.tau      = None

    def calcular(self, mat_path='brt_result.mat', **kwargs):
        print(f"Cargando BRT desde {mat_path}...")

        with h5py.File(mat_path, 'r') as f:
            data_raw  = f['data'][:]         # (T, Ntheta, Ny, Nx)
            tau_raw   = f['tau2'][:]
            g_min     = np.array(f['g_export']['min']).flatten()
            g_max     = np.array(f['g_export']['max']).flatten()
            g_N       = np.array(f['g_export']['N']).flatten().astype(int)
            dVdx_raw  = f['dVdx'][:]         # (Ntheta, Ny, Nx)
            dVdy_raw  = f['dVdy'][:]         # (Ntheta, Ny, Nx)
            dVdt_raw  = f['dVdtheta'][:]     # (Ntheta, Ny, Nx)

        # h5py transpone MATLAB → reordenar a (T, Nx, Ny, Ntheta)
        data     = np.transpose(data_raw, (0, 3, 2, 1))
        # Gradientes: (Ntheta, Ny, Nx) → (Nx, Ny, Ntheta)
        dVdx     = np.transpose(dVdx_raw, (2, 1, 0))
        dVdy     = np.transpose(dVdy_raw, (2, 1, 0))
        dVdtheta = np.transpose(dVdt_raw, (2, 1, 0))

        Nx, Ny, Ntheta = g_N[0], g_N[1], g_N[2]

        self.xs       = np.linspace(g_min[0], g_max[0], Nx)
        self.ys       = np.linspace(g_min[1], g_max[1], Ny)
        self.thetas   = np.linspace(g_min[2], g_max[2], Ntheta)
        self.tau      = tau_raw.flatten()
        self.data     = data
        self.dVdx     = dVdx
        self.dVdy     = dVdy
        self.dVdtheta = dVdtheta

        print(f"  data shape:     {data.shape}")
        print(f"  dVdx shape:     {dVdx.shape}")
        print(f"  dVdy shape:     {dVdy.shape}")
        print(f"  dVdtheta shape: {dVdtheta.shape}")
        print(f"  val t=0:   min={data[0].min():.3f}, max={data[0].max():.3f}")
        print(f"  val t=end: min={data[-1].min():.3f}, max={data[-1].max():.3f}")
        print("BRT cargado correctamente.")

        return self.xs, data

    def _interpolar_estado(self, x, y, theta_rad):
        """Interpola índices en la grilla para un estado dado."""
        Nx, Ny, Ntheta = len(self.xs), len(self.ys), len(self.thetas)
        fi = (x - self.xs[0]) / (self.xs[-1] - self.xs[0]) * (Nx - 1)
        fj = (y - self.ys[0]) / (self.ys[-1] - self.ys[0]) * (Ny - 1)
        fk = (theta_rad - self.thetas[0]) / (self.thetas[-1] - self.thetas[0]) * (Ntheta - 1)
        i = int(np.clip(np.round(fi), 0, Nx - 1))
        j = int(np.clip(np.round(fj), 0, Ny - 1))
        k = int(np.clip(np.round(fk), 0, Ntheta - 1))
        return i, j, k

    def obtener_corte(self, theta_deg):
        """Retorna el corte 2D del BRT para un ángulo dado."""
        theta_rad = np.radians(theta_deg)
        idx = int(np.argmin(np.abs(self.thetas - theta_rad)))
        idx = np.clip(idx, 0, len(self.thetas) - 1)

        corte = self.data[-1, :, :, idx]

        print(f"=== Corte θ={theta_deg}° (idx={idx}, θ_real={np.degrees(self.thetas[idx]):.1f}°) ===")
        print(f"  min={corte.min():.3f}, max={corte.max():.3f}")
        print(f"  % negativo (peligroso): {(corte < 0).mean()*100:.1f}%")
        print("====================================")

        return self.xs, self.ys, corte

    def obtener_control_lrf(self, x, y, theta_deg):
        """
        Control LRF (Least Restrictive Filter) — antes llamado bang-bang.
        Aplica w = ±w_max según el signo de ∂V/∂θ.
        El main.py decide cuándo activarlo según el umbral configurado.
        """
        theta_rad = np.radians(theta_deg)
        i, j, k  = self._interpolar_estado(x, y, theta_rad)

        V    = float(self.data[-1, i, j, k])
        dVdt = float(self.dVdtheta[i, j, k])

        # w* = w_max * sign(∂V/∂θ)
        w = self.w_max * np.sign(dVdt) if dVdt != 0 else 0.0

        return {
            "V":         round(V, 4),
            "dVdtheta":  round(dVdt, 4),
            "w":         round(float(w), 4),
            "peligroso": V < 0
        }

    def obtener_control_cbf(self, x, y, theta_deg, w_usuario, alpha=1.0):
        """
        Control CBF (Control Barrier Function).
        Encuentra el w más cercano al del usuario que satisface:
            ∂V/∂x·v·cos(θ) + ∂V/∂y·v·sin(θ) + ∂V/∂θ·w + α·V ≥ 0

        Parámetros:
            w_usuario: giro deseado por el usuario (rad/s)
            alpha:     parámetro de agresividad del CBF (> 0)
        """
        theta_rad = np.radians(theta_deg)
        i, j, k  = self._interpolar_estado(x, y, theta_rad)

        V       = float(self.data[-1, i, j, k])
        dVdx    = float(self.dVdx[i, j, k])
        dVdy    = float(self.dVdy[i, j, k])
        dVdtheta = float(self.dVdtheta[i, j, k])

        # Término libre de la restricción CBF (todo menos ∂V/∂θ·w)
        Lf_V = dVdx * self.speed * np.cos(theta_rad) + \
               dVdy * self.speed * np.sin(theta_rad)

        # Restricción: dVdtheta·w ≥ -Lf_V - alpha·V
        rhs = -Lf_V - alpha * V

        # Si dVdtheta ≈ 0, no podemos controlar V con w
        if abs(dVdtheta) < 1e-6:
            w_cbf = w_usuario
        else:
            # w mínimo necesario para satisfacer la restricción
            w_min_necesario = rhs / dVdtheta

            # Proyectar w_usuario al conjunto factible
            if dVdtheta > 0:
                # Restricción: w ≥ w_min_necesario
                w_cbf = max(w_usuario, w_min_necesario)
            else:
                # Restricción: w ≤ w_min_necesario
                w_cbf = min(w_usuario, w_min_necesario)

        # Clampear al rango permitido
        w_cbf = float(np.clip(w_cbf, -self.w_max, self.w_max))

        return {
            "V":          round(V, 4),
            "dVdtheta":   round(dVdtheta, 4),
            "w_usuario":  round(float(w_usuario), 4),
            "w":          round(w_cbf, 4),
            "peligroso":  V < 0,
            "intervenido": abs(w_cbf - w_usuario) > 1e-4
        }

    # Mantener compatibilidad con código anterior
    def obtener_control_bangbang(self, x, y, theta_deg):
        return self.obtener_control_lrf(x, y, theta_deg)
