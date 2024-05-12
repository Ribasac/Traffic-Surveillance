This project utilizes state-of-the-art object detection and tracking techniques to perform traffic surveillance. The combination of YOLOv8 for object detection and DeepSORT for object tracking enables accurate and efficient monitoring of vehicles in real-time video streams.

![r](https://github.com/Ribasac/Traffic-Surveillance/assets/6942493/69cfb9c7-e9fb-4a95-8607-fe4dda3f72a3)

# Features
- Object Detection with YOLOv8: YOLOv8 (You Only Look Once version 8) is employed for real-time object detection. It efficiently detects vehicles in video frames with high accuracy.
- Object Tracking with DeepSORT: DeepSORT (Deep Simple Online and Realtime Tracking) is used for robust object tracking across video frames. It associates detected vehicles over consecutive frames to maintain consistent tracking identities.
- Traffic Flow Analysis: The system provides insights into traffic flow patterns, including vehicle count, speed estimation, and trajectory analysis.

# Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- YOLOv8 weights and configuration files
- DeepSORT implementation
