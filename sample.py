import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO

from hazard_scoring import (
    score_detections,
    detections_from_yolo_result
)


st.set_page_config(
    page_title="Hazard Detection",
    page_icon="⚠️"
)

st.title("⚠️ Real-Time Hazard Detection")

model = YOLO("best.pt")


class HazardDetector(VideoProcessorBase):

    def recv(self, frame):

        # Convert incoming browser frame to OpenCV
        img = frame.to_ndarray(format="bgr24")

        # YOLO detection
        results = model(
            img,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes
        annotated = result.plot()

        # Hazard scoring
        detections = detections_from_yolo_result(result)
        score_info = score_detections(detections)

        risk_level = score_info["risk_level"]
        score = score_info["composite_score"]

        # Risk colour
        color = {
            "LOW": (0, 200, 0),
            "MODERATE": (0, 165, 255),
            "CRITICAL": (0, 0, 255)
        }.get(
            risk_level,
            (255, 255, 255)
        )

        # Put risk information on video
        label = f"Risk: {risk_level} | Score: {score:.1f}"

        cv2.putText(
            annotated,
            label,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


webrtc_streamer(
    key="hazard-detection",
    video_processor_factory=HazardDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
