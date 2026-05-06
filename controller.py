class Controller:
    """Traduce las teclas del teclado en comandos de velocidad para el robot."""

    def __init__(self, v_fija=1.0, w_max=1.5):
        self.v_fija  = v_fija
        self.w_max   = w_max
        self.v       = 0.0
        self.w       = 0.0
        self.activo  = False

    def set_velocidad(self, valor):
        """Actualiza la velocidad fija desde el slider o cajón numérico."""
        self.v_fija = max(0.1, min(5.0, valor))

    def set_giro(self, w):
        """Establece el giro directamente desde el frontend."""
        self.w = w

    def activar(self):
        """Activa el movimiento solo si el robot está detenido."""
        if not self.activo:
            self.activo = True

    def detener(self):
        """Detiene completamente el robot."""
        self.activo = False
        self.v      = 0.0
        self.w      = 0.0

    def reset(self):
        """Reinicia el controlador al estado inicial."""
        self.v      = 0.0
        self.w      = 0.0
        self.activo = False

    def obtener_comandos(self):
        """Devuelve los valores actuales de velocidad."""
        if self.activo:
            self.v = self.v_fija
        else:
            self.v = 0.0
        return round(self.v, 3), round(self.w, 3)