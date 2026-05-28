import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

class DubinsHJR:
    """
    Calcula conjuntos alcanzables para un Dubins Car
    usando Hamilton-Jacobi Reachability.
    Estado: (x, y, theta)
    """

    def __init__(self, v=1.0, w_max=1.5):
        self.v      = v
        self.w_max  = w_max
        self.grid   = None
        self.values = None
        self.tau    = None

    def calcular(self,
                 grid_min=(-25.0, -25.0, -np.pi),
                 grid_max=( 25.0,  25.0,  np.pi),
                 N=(31, 31, 13),
                 t_max=1.0,
                 dt=0.1,
                 modo="backward"):

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                lo=jnp.array(grid_min),
                hi=jnp.array(grid_max)
            ),
            N,
            periodic_dims=2
        )

        values = self._funcion_inicial(grid)
        tau    = np.arange(0, t_max + dt, dt)

        dynamics = DubinsCarDynamics(v=self.v, w_max=self.w_max)

        solver_settings = hj.SolverSettings.with_accuracy(
            "medium"
        )

        result = hj.solve(
            solver_settings,
            dynamics,
            grid,
            tau,
            values,
            progress_bar=False
        )

        self.grid   = grid
        self.values = result
        self.tau    = tau

        return grid, result

    def _funcion_inicial(self, grid):
        x = grid.states[..., 0]
        y = grid.states[..., 1]

        # SDF del muro perimetral
        dist_muro = jnp.minimum(
            jnp.minimum(x - (-25.0), 25.0 - x),
            jnp.minimum(y - (-25.0), 25.0 - y)
        )
        values = dist_muro

        # Obstáculos rectangulares — SDF negativo dentro
        obstaculos_rect = [
            {"x":  10.0, "y": 14.0,  "w": 12.0, "h": 4.0},
            {"x": -5.0,  "y": -10.0, "w": 4.0,  "h": 8.0},
            {"x":  10.0, "y": -10.0, "w": 8.0,  "h": 4.0},
        ]

        for obs in obstaculos_rect:
            dx     = jnp.abs(x - obs["x"]) - obs["w"] / 2
            dy     = jnp.abs(y - obs["y"]) - obs["h"] / 2
            dist   = jnp.maximum(dx, dy)
            values = jnp.minimum(values, dist)

        # Obstáculo circular — SDF negativo dentro
        obstaculos_circ = [
            {"x": -14.0, "y": 12.0, "r": 2.5},
        ]

        for obs in obstaculos_circ:
            dist   = jnp.sqrt((x - obs["x"])**2 + (y - obs["y"])**2) - obs["r"]
            values = jnp.minimum(values, dist)

        return values

    def obtener_corte(self, theta_deg):
        theta_rad = np.radians(theta_deg)
        N_theta   = self.grid.shape[2]
        theta_min = -np.pi
        theta_max =  np.pi
        idx = int((theta_rad - theta_min) / (theta_max - theta_min) * (N_theta - 1))
        idx = np.clip(idx, 0, N_theta - 1)

        corte = np.array(self.values[-1, :, :, idx])
        xs    = np.array(self.grid.states[:, 0, 0, 0])
        ys    = np.array(self.grid.states[0, :, 0, 1])

        return xs, ys, corte


class DubinsCarDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """
    Sistema dinámico del Dubins Car.
    dx/dt     = v * cos(theta)
    dy/dt     = v * sin(theta)
    dtheta/dt = w
    """

    control_mode     = "min"
    disturbance_mode = "max"

    def __init__(self, v=1.0, w_max=1.5):
        self.v     = v
        self.w_max = w_max

    def open_loop_dynamics(self, state, time):
        x, y, theta = state[..., 0], state[..., 1], state[..., 2]
        return jnp.stack([
            self.v * jnp.cos(theta),
            self.v * jnp.sin(theta),
            jnp.zeros_like(theta)
        ], axis=-1)

    def control_jacobian(self, state, time):
        n = state.shape[:-1]
        return jnp.stack([
            jnp.zeros(n),
            jnp.zeros(n),
            jnp.ones(n)
        ], axis=-1)[..., jnp.newaxis]

    def disturbance_jacobian(self, state, time):
        n = state.shape[:-1]
        return jnp.zeros(n + (3, 1))

    @property
    def control_space(self):
        return hj.sets.Box(
            lo=jnp.array([-self.w_max]),
            hi=jnp.array([ self.w_max])
        )

    @property
    def disturbance_space(self):
        return hj.sets.Box(
            lo=jnp.array([0.0]),
            hi=jnp.array([0.0])
        )