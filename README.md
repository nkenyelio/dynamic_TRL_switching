Dynamic Transfer Learning Switching on Edge Devices
Overview
This project implements an AI-driven transfer learning switching mechanism for edge devices, enabling them to dynamically switch between pre-trained models based on real-time performance metrics (latency, accuracy, resource availability). This approach ensures efficient inference across heterogeneous edge environments while adapting to changing conditions.

Features
* Dynamic Model Switching – Automatically selects the best pre-trained model at runtime.
* Real-Time Performance Monitoring – Tracks latency, accuracy, and device constraints.
* Unaligned Model Support – Handles different architectures without retraining from scratch.
- Edge-Optimized Deployment – Runs on resource-constrained IoT/MEC devices.

- Installation & Setup
1. Clone the Repository
   git clone https://github.com/nkenyelio/dynamic_TRL_switching.git
   cd dynamic_TRL_switching
2. Install Dependencies
pip install tensorflow torch torchvision numpy psutil
  
How It Works
Edge device collects real-time inference performance (e.g., response time, CPU/memory usage).
Decision engine selects the best available model based on predefined thresholds.
Example Use Case
A smart camera at an edge location can dynamically switch between lightweight (low latency) and heavy (high accuracy) models based on network conditions and hardware capabilitie
