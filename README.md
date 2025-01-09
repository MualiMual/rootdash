Real-Time Inference Camera Feed for Environmental Monitoring

This project is designed to leverage real-time inference using camera feeds to monitor and analyze environmental conditions and the growth of natural elements such as bugs, birds, and plants. The system integrates advanced machine learning capabilities using Coral Edge TPU to process data at the edge, providing efficient and fast analytics.

Data Flow
Real-time Inference: Camera feeds for detecting and analyzing bugs, birds, and plants.
Images Processing: Analyzes images of plants focusing on leaves, stems, and flowers.
Environmental Monitoring: Collects data on temperature and humidity from integrated sensors.
Growth Tracking: Employs object detection techniques to measure plant size and monitor growth rates.

Key Features
Coral Edge TPU Integration: Utilizes Coral Edge TPU for accelerated machine learning tasks.
Model Management: Organized approach for managing different ML models, focusing on growth analysis and object detection.
Sensor Integration: Robust monitoring of environmental data crucial for real-time processing.
Web Dashboard: A web interface for data visualization and interaction with the models.

Directory Structure
/dashboard
│
├── models/
│   ├── growth_analysis.py
│   ├── object_detection.py
│   └── models.py
│
├── pycoral/
│   ├── benchmarks/
│   ├── build/
│   ├── docs/
│   ├── examples/
│   ├── libcoral/
│   ├── libedgetpu/
│   └── src/
│
├── sensors/
│   ├── sensor_reader.py
│   └── system_monitor.py
│
├── server.py
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── test_data/
│   ├── automl_video_ondevice/
│   ├── imprinting/
│   ├── pipeline/
│   └── posenet/
│
└── utils/
    ├── conversions.py
    ├── network.py
    └── camera.py

Author: Xavier J Bass @MualiMual MualiMual.com
