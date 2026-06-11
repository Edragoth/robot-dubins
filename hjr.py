import numpy as np
import h5py
 
 
class DubinsHJR:
    """
    Carga el BRT pre-calculado desde MATLAB (brt_result.mat)
    y lo sirve para visualización.
    """
 
    def __init__(self, speed=5.0, w_max=1.3):
        self.speed  = speed
        self.w_max  = w_max
        self.data   = None
        self.xs     = None
        self.ys     = None
        self.thetas = None
        self.tau    = None
 
    def calcular(self, mat_path='brt_result.mat', **kwargs):
        print(f"Cargando BRT desde {mat_path}...")
 
        with h5py.File(mat_path, 'r') as f:
            # Imprimir estructura para diagnóstico
            print("  Claves en el archivo:")
            def print_keys(name, obj):
                print(f"    {name}: shape={getattr(obj, 'shape', 'grupo')}")
            f.visititems(print_keys)
 
            # Cargar data y tau
            data_raw = f['data'][:]    # MATLAB (Nx,Ny,Ntheta,T) → h5py (T,Ntheta,Ny,Nx)
            tau_raw  = f['tau2'][:]
 
            # Cargar parámetros de grilla — manejar distintas formas
            g_min = np.array(f['g_export']['min']).flatten()
            g_max = np.array(f['g_export']['max']).flatten()
            g_N   = np.array(f['g_export']['N']).flatten().astype(int)
 
        print(f"  data_raw shape: {data_raw.shape}")
        print(f"  g_min: {g_min}")
        print(f"  g_max: {g_max}")
        print(f"  g_N:   {g_N}")
 
        # h5py transpone MATLAB: reordenar a (T, Nx, Ny, Ntheta)
        data = np.transpose(data_raw, (0, 3, 2, 1))
 
        Nx, Ny, Ntheta = g_N[0], g_N[1], g_N[2]
 
        self.xs     = np.linspace(g_min[0], g_max[0], Nx)
        self.ys     = np.linspace(g_min[1], g_max[1], Ny)
        self.thetas = np.linspace(g_min[2], g_max[2], Ntheta)
        self.tau    = tau_raw.flatten()
        self.data   = data
 
        print(f"  data shape final: {data.shape}")
        print(f"  val t=0:   min={data[0].min():.3f}, max={data[0].max():.3f}")
        print(f"  val t=end: min={data[-1].min():.3f}, max={data[-1].max():.3f}")
        print("BRT cargado correctamente.")
 
        return self.xs, data
 
    def obtener_corte(self, theta_deg):
        theta_rad = np.radians(theta_deg)
        idx = int(np.argmin(np.abs(self.thetas - theta_rad)))
        idx = np.clip(idx, 0, len(self.thetas) - 1)
 
        corte = self.data[-1, :, :, idx]
 
        print(f"=== Corte θ={theta_deg}° (idx={idx}, θ_real={np.degrees(self.thetas[idx]):.1f}°) ===")
        print(f"  min={corte.min():.3f}, max={corte.max():.3f}")
        print(f"  % negativo (peligroso): {(corte < 0).mean()*100:.1f}%")
        print("====================================")
 
        return self.xs, self.ys, corte
