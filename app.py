import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
from dubins_car import DubinsCar
from planner import DubinsPlanner
from controller import Controller

st.set_page_config(page_title="Robot Dubins", layout="wide")
st.title("Control de Robot Dubins Car")

if "robot" not in st.session_state:
    st.session_state.robot       = DubinsCar(x=0.0, y=0.0, theta=0.0)
    st.session_state.planner     = DubinsPlanner(radio_min=0.5)
    st.session_state.controller  = Controller()
    st.session_state.trayectoria = [(0.0, 0.0)]
    st.session_state.modo        = "manual"
    st.session_state.tecla       = ""

st.sidebar.title("Configuración")
modo = st.sidebar.radio("Modo de control", ["manual", "automático"])
st.session_state.modo = modo

if st.sidebar.button("Reiniciar robot"):
    st.session_state.robot.reset(x=0.0, y=0.0, theta=0.0)
    st.session_state.controller.reset()
    st.session_state.trayectoria = [(0.0, 0.0)]

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado del robot**")
estado = st.session_state.robot.obtener_estado()
st.sidebar.write(f"X:      {estado['x']}")
st.sidebar.write(f"Y:      {estado['y']}")
st.sidebar.write(f"Ángulo: {round(estado['theta'] * 180 / 3.14159, 1)}°")
st.sidebar.write(f"V:      {estado['v']} m/s")
st.sidebar.write(f"W:      {estado['w']} rad/s")

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (metros)")
    ax.set_ylabel("Y (metros)")
    ax.set_title("Mapa del robot")

    if len(st.session_state.trayectoria) > 1:
        xs = [p[0] for p in st.session_state.trayectoria]
        ys = [p[1] for p in st.session_state.trayectoria]
        ax.plot(xs, ys, color="green", linewidth=1.5, label="Trayectoria")

    x     = estado["x"]
    y     = estado["y"]
    theta = estado["theta"]
    ax.annotate("", xy=(x + 0.5 * np.cos(theta), y + 0.5 * np.sin(theta)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.plot(x, y, "bo", markersize=10, label="Robot")
    ax.legend()
    st.pyplot(fig)

with col2:
    if modo == "manual":
        st.markdown("### Control manual")

        components.html("""
        <div style="text-align:center; font-family:sans-serif;">
            <p style="font-size:13px; color:gray;">Presiona las flechas del teclado</p>
            <div id="key-up"    style="display:inline-block; padding:10px 18px; m