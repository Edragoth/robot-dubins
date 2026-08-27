from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dubins_car import DubinsCar
from planner import DubinsPlanner
from controller import Controller
from hjr import DubinsHJR
from apf import APF
import json
import math
import asyncio
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

robot       = DubinsCar(x=0.0, y=0.0, theta=0.0)
planner     = DubinsPlanner(radio_min=0.5)
controller  = Controller()
trayectoria = [(0.0, 0.0)]

LIMITE = 20.0

# Umbrales de seguridad por modo de control
# Ajustar independientemente para cada método
UMBRALES = {
    'manual': 0.0,   # sin control — no interviene
    'lrf':    2.0,   # LRF (Least Restrictive Filter)
    'cbf':    2.0,   # CBF (Control Barrier Function)
    'apf':    3.0,   # APF (Artificial Potential Field)
}

# Parámetro alpha del CBF
CBF_ALPHA = 1.0

obstaculos_rect = [
    {"x":  10.0, "y": 14.0,  "w": 12.0, "h": 4.0,  "ang": 0},
    {"x":  -5.0, "y": -10.0, "w":  4.0, "h": 8.0,  "ang": 0},
    {"x":  10.0, "y": -10.0, "w":  8.0, "h": 4.0,  "ang": 0},
]

obstaculos_circ = [
    {"x": -14.0, "y": 12.0, "r": 2.5},
]

hjr_instance   = None
hjr_listo      = False
hjr_calculando = False
apf_instance   = APF(speed=5.0, w_max=1.3, eta=1.0, d0=5.0)

# Modo de control activo
modo_control = 'manual'

def punto_en_rectangulo(px, py, obs):
    ang = math.radians(obs["ang"])
    dx  = px - obs["x"]
    dy  = py - obs["y"]
    lx  =  dx * math.cos(ang) + dy * math.sin(ang)
    ly  = -dx * math.sin(ang) + dy * math.cos(ang)
    return abs(lx) <= obs["w"] / 2 and abs(ly) <= obs["h"] / 2

def punto_en_circulo(px, py, obs):
    dx = px - obs["x"]
    dy = py - obs["y"]
    return math.sqrt(dx**2 + dy**2) <= obs["r"]

def verificar_colision(x, y):
    if abs(x) >= LIMITE or abs(y) >= LIMITE:
        return True
    for obs in obstaculos_rect:
        if punto_en_rectangulo(x, y, obs):
            return True
    for obs in obstaculos_circ:
        if punto_en_circulo(x, y, obs):
            return True
    return False

def precalcular_hjr():
    global hjr_instance, hjr_listo, hjr_calculando
    print("Iniciando precálculo Hamilton-Jacobi...")
    hjr_calculando = True
    try:
        hjr = DubinsHJR(speed=5.0, w_max=1.3)
        hjr.calcular(mat_path='brt_result.mat')
        hjr_instance  = hjr
        hjr_listo     = True
        print("Precálculo HJR completado.")
    except Exception as e:
        print(f"Error en precálculo HJR: {e}")
    hjr_calculando = False

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, precalcular_hjr)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/hjr_estado")
async def hjr_estado():
    return {
        "listo":      hjr_listo,
        "calculando": hjr_calculando
    }

@app.get("/hjr")
async def calcular_hjr(theta: float = 0.0, v: float = 1.0, modo: str = "backward"):
    if not hjr_listo:
        return {"error": "HJR aún calculando, espera un momento"}
    xs, ys, corte = hjr_instance.obtener_corte(theta)
    return {
        "xs":    xs.tolist(),
        "ys":    ys.tolist(),
        "corte": corte.tolist(),
        "modo":  modo,
        "theta": theta
    }

@app.get("/control")
async def obtener_control(x: float = 0.0, y: float = 0.0, theta: float = 0.0, modo: str = "lrf", w_usuario: float = 0.0):
    """Retorna el control de seguridad para la posición dada."""
    if modo == "apf":
        return apf_instance.obtener_control_apf(x, y, theta, w_usuario)
    if not hjr_listo:
        return {"error": "HJR aún calculando"}
    if modo == "cbf":
        return hjr_instance.obtener_control_cbf(x, y, theta, w_usuario, alpha=CBF_ALPHA)
    return hjr_instance.obtener_control_lrf(x, y, theta)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global trayectoria, modo_control
    await websocket.accept()
    try:
        while True:
            data    = await websocket.receive_text()
            mensaje = json.loads(data)

            if mensaje["tipo"] == "tecla":
                if mensaje["tecla"] == "ArrowUp":
                    controller.activar()
                elif mensaje["tecla"] == "ArrowDown":
                    controller.detener()

            elif mensaje["tipo"] == "giro":
                controller.set_giro(mensaje["valor"])

            elif mensaje["tipo"] == "velocidad":
                controller.set_velocidad(mensaje["valor"])

            elif mensaje["tipo"] == "modo_control":
                modo_control = mensaje["valor"]
                print(f"Modo control: {modo_control}")

            elif mensaje["tipo"] == "reset":
                robot.reset(x=0.0, y=0.0, theta=0.0)
                controller.reset()
                trayectoria = [(0.0, 0.0)]

            elif mensaje["tipo"] == "automatico":
                tray = planner.planificar(
                    robot.x, robot.y, robot.theta,
                    mensaje["x_fin"], mensaje["y_fin"], mensaje["theta_fin"]
                )
                trayectoria = tray
                ultimo      = tray[-1]
                robot.reset(x=ultimo[0], y=ultimo[1], theta=mensaje["theta_fin"])

            elif mensaje["tipo"] == "tick":
                pass

            v, w = controller.obtener_comandos()

            # Aplicar control según modo activo
            control_info = {"V": 0.0, "peligroso": False, "w": w, "intervenido": False}

            if controller.activo and modo_control != 'manual':
                theta_deg = math.degrees(robot.theta)
                umbral    = UMBRALES.get(modo_control, 0.0)

                if modo_control == 'lrf' and hjr_listo:
                    resultado = hjr_instance.obtener_control_lrf(
                        robot.x, robot.y, theta_deg
                    )
                    control_info = resultado
                    if resultado["V"] < umbral:
                        w = resultado["w"]
                        control_info["intervenido"] = True

                elif modo_control == 'cbf' and hjr_listo:
                    resultado = hjr_instance.obtener_control_cbf(
                        robot.x, robot.y, theta_deg, w, alpha=CBF_ALPHA
                    )
                    control_info = resultado
                    if resultado["V"] < umbral:
                        w = resultado["w"]

                elif modo_control == 'apf':
                    resultado = apf_instance.obtener_control_apf(
                        robot.x, robot.y, theta_deg, w
                    )
                    control_info = resultado
                    # APF siempre aplica — la mezcla gradual es interna al apf.py
                    w = resultado["w"]
                    print(f"APF: x={robot.x:.1f}, y={robot.y:.1f}, w_usr={w:.2f}, w_apf={resultado['w']:.2f}, f_mag={resultado['f_mag']:.3f}, V={resultado['V']:.2f}")

            elif hjr_listo:
                # Solo calcular V para mostrarlo, sin intervenir
                theta_deg = math.degrees(robot.theta)
                resultado = hjr_instance.obtener_control_lrf(robot.x, robot.y, theta_deg)
                control_info["V"] = resultado["V"]

            robot.actualizar(v, w, dt=0.1)

            if verificar_colision(robot.x, robot.y):
                robot.reset(x=0.0, y=0.0, theta=0.0)
                controller.reset()
                trayectoria = [(0.0, 0.0)]

            trayectoria.append((robot.x, robot.y))

            umbral = UMBRALES.get(modo_control, 0.0)

            estado = robot.obtener_estado()
            estado["trayectoria"]  = trayectoria[-300:]
            estado["colision"]     = False
            estado["activo"]       = controller.activo
            estado["hjr_listo"]    = hjr_listo
            estado["peligroso"]    = control_info.get("V", 0.0) < umbral
            estado["w_control"]    = control_info.get("w", 0.0)
            estado["V"]            = control_info.get("V", 0.0)
            estado["intervenido"]  = control_info.get("intervenido", False)
            estado["modo_control"] = modo_control

            await websocket.send_text(json.dumps(estado))

    except Exception as e:
        print(f"Conexión cerrada: {e}")
