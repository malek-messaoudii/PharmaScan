import threading

class SystemState:
    def __init__(self):
        self.latest_result = "En attente du résultat de détection"
        self.mqtt_client = None
        self.lock = threading.Lock()

system_state = SystemState()
