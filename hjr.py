import numpy as np
import h5py


class DubinsHJR:
    """
    Carga el BRT pre-calculado desde MATLAB (brt_result.mat)
    y lo sirve para visualización y control bang-bang.
    """

    def __init__(self, speed=5.0, w_max=1.3):
        self.speed    = speed
        self.w_max    = w_max
        self.data     = None   # shape: (T, Nx, Ny, Ntheta)
        self.dVdtheta = None   # shape: (Nx, Ny, Ntheta)
        self.xs       = None
        self.ys       = None
        self.thetas   = None
        self.tau      = None

    def calcular(self, mat_path='brt_result.mat', **kwargs):
        print(f"Cargando BRT desde {mat_path}...")

        with h5py.File(mat_path, 'r') as f:
            data_raw  = f['data'][:]       # (T, Ntheta, Ny, Nx)
            tau_raw   = f['tau2'][:]
            g_min     = np.array(f['g_export']['min']).flatten()
            g_max     = np.array(f['g_export']['max']).flatten()
            g_N       = np.array(f['g_export']['N']).flatten().astype(int)
            dVdt_raw  = f['dVdtheta'][:]   # (Ntheta, Ny, Nx)

        # h5py transpone MATLAB → reordenar a (T, Nx, Ny, Ntheta)
        data     = np.transpose(data_raw, (0, 3, 2, 1))
        # dVdtheta: (Ntheta, Ny, Nx) → (Nx, Ny, Ntheta)
        dVdtheta = np.transpose(dVdt_raw, (2, 1, 0))

        Nx, Ny, Ntheta = g_N[0], g_N[1], g_N[2]

        self.xs       = np.linspace(g_min[0], g_max[0], Nx)
        self.ys       = np.linspace(g_min[1], g_max[1], Ny)
        self.thetas   = np.linspace(g_min[2], g_max[2], Ntheta)
        self.tau      = tau_raw.flatten()
        self.data     = data
        self.dVdtheta = dVdtheta

        print(f"  data shape:     {data.shape}")
        print(f"  dVdtheta shape: {dVdtheta.shape}")
        print(f"  val t=0:   min={data[0].min():.3f}, max={data[0].max():.3f}")
        print(f"  val t=end: min={data[-1].min():.3f}, max={data[-1].max():.3f}")
        print("BRT cargado correctamente.")

        return self.xs, data

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

    def obtener_control_bangbang(self, x, y, theta_deg):
        """
        Calcula el control bang-bang en la posición (x, y, theta).
        Retorna w = +w_max o -w_max según el signo de dV/dtheta.
        Solo se aplica cuando el robot está en zona peligrosa (V < 0).
        """
        theta_rad = np.radians(theta_deg)

        # Índices en la grilla
        Nx, Ny, Ntheta = len(self.xs), len(self.ys), len(self.thetas)

        fi = (x - self.xs[0]) / (self.xs[-1] - self.xs[0]) * (Nx - 1)
        fj = (y - self.ys[0]) / (self.ys[-1] - self.ys[0]) * (Ny - 1)
        fk = (theta_rad - self.thetas[0]) / (self.thetas[-1] - self.thetas[0]) * (Ntheta - 1)

        i = int(np.clip(np.round(fi), 0, Nx - 1))
        j = int(np.clip(np.round(fj), 0, Ny - 1))
        k = int(np.clip(np.round(fk), 0, Ntheta - 1))

        # Valor BRT en esa posición
        V = float(self.data[-1, i, j, k])

        # Gradiente dV/dtheta
        dVdt = float(self.dVdtheta[i, j, k])

        # Control bang-bang: w* = w_max * sign(dV/dtheta)
        # Con uMode='max', el control óptimo maximiza dV/dt
        w = self.w_max * np.sign(dVdt) if dVdt != 0 else 0.0

        return {
            "V":    round(V, 4),
            "dVdt": round(dVdt, 4),
            "w":    round(w, 4),
            "peligroso": V < 0
        }
