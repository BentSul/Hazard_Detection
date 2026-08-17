from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from hazard_scoring import score_detections, detections_from_yolo_result

# Load the trained YOLO model
model = YOLO("C:\\Users\\User\\Desktop\\best.pt")

# --- Single Image Detection and Scoring (Colab-friendly) ---
# IMPORTANT: Ensure 'test_18.jpg' is uploaded to the Colab environment
# via the upload cell (69d2e7c4) for this code to work.

# image_path = "C:\\Users\\User\\Desktop\\test_a.jpg"
# image = cv2.imread(image_path)

# if image is None:
#     print(f"ERROR: Could not load image from {image_path}. Please ensure it is uploaded.")
# else:
#     print(f"Processing image: {image_path}")
#     results = model(image, verbose=False) # verbose=False to suppress detailed output per frame
#     annotated_frame = results[0].plot() # Get annotated frame for display

#     # --- Scoring integration ---
#     detections = detections_from_yolo_result(results[0])
#     score_info = score_detections(detections)

#     # Overlay the risk level + score onto the frame
#     label = f"Risk: {score_info['risk_level']} (score: {score_info['composite_score']:.1f})"
#     color = {
#         "LOW": (0, 200, 0),
#         "MODERATE": (0, 165, 255),
#         "CRITICAL": (0, 0, 255),
#     }.get(score_info["risk_level"], (255, 255, 255)) # Default to white if risk level is unknown

#     cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # Display the annotated frame
#     plt.figure(figsize=(12, 8))
#     plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
#     print(f"\nScoring results: {score_info}")

# --- Real-time detection (not directly supported in Colab's interactive environment) ---
# The following code is for reference if you were running this on a local machine
# with a webcam. In Colab, direct cv2.VideoCapture(0) and cv2.imshow are problematic.
# For video processing in Colab, you would typically upload a video file and process
# it frame by frame, or use specific Colab webcam integration libraries if available.
#
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam feed started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        detections = detections_from_yolo_result(results[0])
        score_info = score_detections(detections)

        label = f"Risk: {score_info['risk_level']} (score: {score_info['composite_score']:.1f})"
        color = {
            "LOW": (0, 200, 0),
            "MODERATE": (0, 165, 255),
            "CRITICAL": (0, 0, 255),
        }.get(score_info["risk_level"], (255, 255, 255))

        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Hazard Detection - Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
