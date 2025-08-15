# PharmaScan – Intelligent Medicine Package Detection & Labeling 

PharmaScan is an AI-powered computer vision pipeline for detecting, localizing, and labeling pharmaceutical medicine packages from images or video streams.
Built with YOLOv8, advanced image enhancement, and precision contour refinement, PharmaScan is designed for accuracy, clarity, and real-world usability in pharmaceutical inventory, counterfeit detection, and automated cataloging.

# Real-Time Medicine Counting — FastAPI + MQTT Integration

PharmaScan isn’t just about image recognition — it’s built to operate in real-time, integrating with FastAPI and MQTT for automated medicine counting in pharmacies, hospitals, or warehouses.

## How It Works : 
1. Start the FastAPI Server
   ```
   python -m uvicorn app.main:app --reload
   ```
   
2. Set Target Quantity via /quantity API
   ```
   curl -X POST "http://localhost:8000/quantity" \-H "Content-Type: application/json" \-d '{ "quantity": 10 }'
   ```
   
3. Camera Detection & Counting
   * The camera continuously captures frames.
   * Best.pt detects medicine boxes in real time.
   * The count increments until it matches your target quantity.
     
4. Auto Stop & Notification
   
    When the quantity is reached:

     * The system automatically stops the camera stream.

     * An MQTT message is published to confirm completion.
