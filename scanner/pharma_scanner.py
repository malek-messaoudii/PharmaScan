import cv2
import numpy as np
import time
import threading
import json
import os
import logging
from datetime import datetime
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from typing import Optional, Set, Dict, List, Any, Tuple
from app.config import (
    MQTT_BROKER, MQTT_PORT, MQTT_CONTROL_TOPIC,
    MQTT_FEEDBACK_TOPIC, MQTT_TOPIC_RESULT, MQTT_TOPIC_STATUS
)
from app.logger import setup_logger

logger = setup_logger(__name__)

class PharmaScanner:
    def __init__(self, camera_index: int = 0):
        """Initialize the medicine box detection system."""
        # Model configuration
        self.model: YOLO = YOLO('BEST.pt')
        self.medicine_class_id: int = 0  # Class ID for medicine_box
        self.confidence_threshold: float = 0.85
        self.min_box_area: int = 4000  # Minimum area for valid detection (pixels)
        
        # Camera configuration
        self.camera_index: int = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_backend: int = self._determine_camera_backend()
        self.frame_size: Tuple[int, int] = (1280, 720)
        
        # Detection tracking
        self.detection_active: bool = False
        self.expected_count: int = 0
        self.detected_ids: Set[int] = set()
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        
        # MQTT client
        self.client: mqtt.Client = mqtt.Client(
            client_id=f"pharmascan-detector-{os.getpid()}",
            protocol=mqtt.MQTTv311
        )
        
        # Threading
        self.lock: threading.Lock = threading.Lock()
        self.detection_thread: Optional[threading.Thread] = None
        self.running: bool = False
        
        # Visualization
        self.window_name: str = "Medicine Box Detection"
        self.colors: Dict[str, Tuple[int, int, int]] = {
            'box': (0, 255, 0),       # Green for medicine boxes
            'text': (0, 0, 255),      # Red for text
            'roi': (255, 255, 0),    # Yellow for ROI
            'status': (0, 0, 255)     # Red for status text
        }

    def _determine_camera_backend(self) -> int:
        """Determine the appropriate camera backend based on the OS."""
        if os.name == 'nt':  # Windows
            return cv2.CAP_DSHOW
        elif os.name == 'posix':  # Linux/macOS
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def start(self) -> None:
        """Start the main application loop."""
        self._setup_mqtt()
        self.running = True
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            self.stop()

    def _setup_mqtt(self) -> None:
        """Configure and start the MQTT client."""
        self.client.on_connect = self._on_mqtt_connect
        self.client.on_message = self._on_mqtt_message
        self.client.on_disconnect = self._on_mqtt_disconnect
        
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            logger.info("MQTT client started successfully")
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")
            raise

    def stop(self) -> None:
        """Clean up resources and stop all components."""
        logger.info("Stopping PharmaScanner...")
        self.running = False
        self.detection_active = False
        
        # Stop detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Release camera
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
        
        # Stop MQTT
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.error(f"Error during MQTT cleanup: {e}")
        
        logger.info("PharmaScanner stopped successfully")

    def _on_mqtt_connect(self, client: mqtt.Client, userdata: Any, flags: Any, rc: int) -> None:
        """Handle MQTT connection events."""
        if rc == 0:
            logger.info("Connected to MQTT broker successfully")
            client.subscribe(MQTT_CONTROL_TOPIC)
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self.stop()

    def _on_mqtt_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Handle MQTT disconnection events."""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (rc: {rc})")
            if self.running:
                self.stop()

    def _on_mqtt_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """Handle incoming MQTT messages."""
        try:
            payload = json.loads(msg.payload.decode())
            logger.debug(f"Received MQTT message: {payload}")
            
            if payload.get("action") == "start":
                self._handle_start_command(payload)
            elif payload.get("action") == "stop":
                self._handle_stop_command()
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def _handle_start_command(self, payload: Dict[str, Any]) -> None:
        """Handle start detection command."""
        with self.lock:
            try:
                self.expected_count = max(0, int(payload.get("expectedCount", 0)))
                if self.expected_count <= 0:
                    logger.warning("Invalid expected count received")
                    return
                
                self.detected_ids.clear()
                self.track_history.clear()
                self.detection_active = True
                
                if not (self.detection_thread and self.detection_thread.is_alive()):
                    self.detection_thread = threading.Thread(
                        target=self._run_detection,
                        daemon=True,
                        name="DetectionThread"
                    )
                    self.detection_thread.start()
                    logger.info(f"Started detection for {self.expected_count} medicine boxes")
                    
            except Exception as e:
                logger.error(f"Error handling start command: {e}")

    def _handle_stop_command(self) -> None:
        """Handle stop detection command."""
        with self.lock:
            self.detection_active = False
            logger.info("Detection stopped by command")

    def _initialize_camera(self) -> bool:
        """Initialize and verify camera connection."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index, self.camera_backend)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            ret, _ = self.cap.read()
            if not ret:
                logger.error("Camera opened but cannot read frames")
                return False
                
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False

    def _process_detection_results(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """Process detection results and annotate the frame."""
        annotated_frame = frame.copy()
        
        # Draw ROI (Region of Interest)
        h, w = frame.shape[:2]
        roi_x1, roi_y1 = int(w * 0.15), int(h * 0.15)
        roi_x2, roi_y2 = int(w * 0.85), int(h * 0.85)
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), self.colors['roi'], 2)
        
        if not results or not hasattr(results[0], 'obb') or results[0].obb is None:
            return annotated_frame
            
        obb = results[0].obb
        boxes = obb.xyxyxyxy.cpu().numpy() if hasattr(obb, 'xyxyxyxy') else []
        classes = obb.cls.int().tolist() if hasattr(obb, 'cls') else []
        confs = obb.conf.tolist() if hasattr(obb, 'conf') else []
        ids = obb.id.int().tolist() if (hasattr(obb, 'id') and obb.id is not None) else []
        
        for i, (box, cls, conf, obj_id) in enumerate(zip(boxes, classes, confs, ids)):
            if cls != self.medicine_class_id or conf < self.confidence_threshold:
                continue
                
            # Convert box to corners
            corners = box.reshape(4, 2).astype(int)
            
            # Check if box is within ROI
            if not all(roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2 for x, y in corners):
                continue
                
            # Calculate box area
            box_area = cv2.contourArea(corners)
            if box_area < self.min_box_area:
                continue
                
            # Draw the oriented bounding box
            for j in range(4):
                p1 = tuple(corners[j])
                p2 = tuple(corners[(j + 1) % 4])
                cv2.line(annotated_frame, p1, p2, self.colors['box'], 2)
            
            # Add label
            label = f"ID:{obj_id} ({conf:.2f})"
            cv2.putText(annotated_frame, label, tuple(corners[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            
            # Update tracking
            with self.lock:
                if obj_id not in self.detected_ids:
                    self.detected_ids.add(obj_id)
                    logger.info(f"Detected medicine box ID {obj_id} ({len(self.detected_ids)}/{self.expected_count})")
                    self._send_status_update()
        
        return annotated_frame

    def _run_detection(self) -> None:
        """Main detection loop."""
        if not self._initialize_camera():
            logger.error("Cannot start detection without camera")
            return
            
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        try:
            while self.detection_active and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame with YOLO
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[self.medicine_class_id],
                    conf=self.confidence_threshold,
                    verbose=False
                )
                
                # Process results and annotate frame
                annotated_frame = self._process_detection_results(frame, results)
                
                # Add status information
                status_text = f"Detected: {len(self.detected_ids)}/{self.expected_count}"
                cv2.putText(annotated_frame, status_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['status'], 2)
                
                # Display the frame
                cv2.imshow(self.window_name, annotated_frame)
                
                # Check for completion or quit command
                if len(self.detected_ids) >= self.expected_count:
                    self._send_result()
                    self.detection_active = False
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
                    
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Detection loop ended")

    def _send_status_update(self) -> None:
        """Send current detection status via MQTT."""
        try:
            payload = {
                "type": "status",
                "detected": len(self.detected_ids),
                "expected": self.expected_count,
                "timestamp": datetime.now().isoformat()
            }
            self.client.publish(MQTT_FEEDBACK_TOPIC, json.dumps(payload))
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")

    def _send_result(self) -> None:
        """Send final detection result via MQTT."""
        try:
            is_complete = len(self.detected_ids) >= self.expected_count
            payload = {
                "type": "result",
                "complete": is_complete,
                "detected": len(self.detected_ids),
                "expected": self.expected_count,
                "timestamp": datetime.now().isoformat()
            }
            self.client.publish(MQTT_TOPIC_RESULT, json.dumps(payload))
            
            if is_complete:
                self.client.publish(MQTT_TOPIC_STATUS, "swipping")
                
            logger.info(f"Detection completed: {len(self.detected_ids)}/{self.expected_count}")
        except Exception as e:
            logger.error(f"Failed to send result: {e}")

if __name__ == "__main__":
    try:
        scanner = PharmaScanner()
        scanner.start()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)