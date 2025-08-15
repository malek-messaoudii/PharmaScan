# app/main.py
from fastapi import FastAPI, HTTPException
from app.logger import setup_logger
from app.models import QuantityRequest
from app.state import system_state
from app.firebase import store_quantity
from app.config import MQTT_TOPIC_QUANTITY, MQTT_TOPIC_STATUS, MQTT_CONTROL_TOPIC
from app.mqtt_client import connect_and_start
from scanner.pharma_scanner import PharmaScanner
import threading
import json

app = FastAPI(title="PharmaScan API")
logger = setup_logger(__name__)

# Start scanner in background thread
scanner = PharmaScanner()
th = threading.Thread(target=scanner.start, daemon=True)
th.start()
logger.info("Scanner thread started")

@app.on_event("startup")
async def on_startup():
    def run_mqtt():
        client = connect_and_start()
        if client:
            system_state.mqtt_client = client
            client.loop_start()
        else:
            logger.error("MQTT client failed to connect")

    threading.Thread(target=run_mqtt, daemon=True).start()
    logger.info("MQTT thread started")

@app.post("/quantity")
async def send_quantity(request: QuantityRequest):
    if request.quantity < 0:
        raise HTTPException(status_code=400, detail="Quantité invalide")

    try:
        # Store in Firebase
        try:
            store_quantity(request.quantity)
            logger.info(f"Quantity {request.quantity} stored in Firebase")
        except Exception as fb_err:
            logger.warning(f"Firebase storage failed: {fb_err}")

        # Publish to MQTT
        if system_state.mqtt_client and system_state.mqtt_client.is_connected():
            system_state.mqtt_client.publish(MQTT_TOPIC_QUANTITY, str(request.quantity))
            system_state.mqtt_client.publish(MQTT_TOPIC_STATUS, "swipping")

            control_payload = json.dumps({
                "action": "start",
                "expectedCount": request.quantity
            })
            system_state.mqtt_client.publish(MQTT_CONTROL_TOPIC, control_payload)

            logger.info(f"Detection started for expectedCount={request.quantity}")
            return {"status": "success", "message": "Détection démarrée"}
        else:
            logger.error("MQTT client not connected")
            raise HTTPException(status_code=503, detail="Client MQTT non connecté")

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result")
async def get_result():
    with system_state.lock:
        return {"result": system_state.latest_result}
