import paho.mqtt.client as mqtt
import json
import time
from app.state import system_state
from app.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC_RESULT
from app.logger import setup_logger

logger = setup_logger(__name__)

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC_RESULT)
        system_state.mqtt_client = client
    else:
        logger.error(f"Failed to connect with code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        logger.info(f"Received message: {payload}")
        with system_state.lock:
            system_state.latest_result = payload
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def connect_and_start():
    client = mqtt.Client(client_id=f"PharmaScan-API-{int(time.time())}")
    client.on_connect = on_connect
    client.on_message = on_message

    for attempt in range(5):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT)
            client.loop_start()
            return
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            time.sleep(2 ** attempt)
