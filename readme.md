# Grasshopper RLHF System

A comprehensive framework for Reinforcement Learning with Human Feedback (RLHF) in Grasshopper 3D modeling environments.

## Overview

This system enables the application of reinforcement learning techniques to parametric architectural designs created in Grasshopper, with a human feedback collection mechanism that helps refine the RL model based on human preferences and design expertise.

The framework consists of:
1. **Reinforcement Learning components** (Python) that interact with Grasshopper models
2. **ZMQ communication layer** (C# + Python) for bidirectional data exchange
3. **Human feedback interface** (Flask web application) for collecting design evaluations
4. **Analysis tools** (Python) for processing and integrating feedback data

## System Architecture

```
┌───────────────┐           ┌───────────────┐          ┌───────────────┐
│  Grasshopper  │◄──ZMQ────►│  Python RL    │◄────────►│  Human        │
│  Environment  │           │  Environment   │          │  Feedback UI  │
└───────────────┘           └───────────────┘          └───────────────┘
    ▲                            ▲                          ▲
    │                            │                          │
    │                            │                          │
    └────────────────────────────┴──────────────────────────┘
                         Data Storage & Analysis
```

## Prerequisites

### Software Requirements
- Rhino 6/7 with Grasshopper
- Python 3.7+ with required packages
- Visual Studio for C# components (.NET Framework 4.8)
- Git for version control

### Installation of Python Dependencies
```bash
pip install flask zmq gymnasium stable-baselines3 torch werkzeug numpy pandas matplotlib seaborn
```

### Installation of C# Components
1. Open the solution in Visual Studio
2. Build the project to generate the DLL
3. Convert the DLL to GHA format using Grasshopper Assembly Utility
4. Place the GHA file in your Grasshopper Components folder

## Project Structure

```
/ (root)
├── data/
│   ├── designs/     # Stored design parameters and metadata
│   ├── feedback/    # Human feedback data
│   ├── meshes/      # 3D mesh exports from Grasshopper
│   └── zmq_logs/    # Communication logs for debugging
│
├── python_modules/
│   ├── ppo_train.py             # PPO training script
│   ├── env_simple.py            # Reinforcement learning environment
│   ├── zmq_state_receiver.py    # Receiver for state and reward data
│   ├── analyze_integrated_data.py # Analysis tools for RLHF data
│   └── design_regenerator.py    # Utility to regenerate optimal designs
│
├── server/
│   ├── app.py                   # Flask server for feedback collection
│   ├── static/                  # Web UI static assets
│   └── templates/               # HTML templates for feedback UI
│
└── src/                         # C# source code
    ├── GrasshopperZmqComponent.sln # Visual Studio solution
    └── GrasshopperZmqComponent/
        ├── SliderInfoExporter.cs # Export slider ranges
        ├── ZmqListener.cs        # Receive actions from Python
        ├── ZmqStateSender.cs     # Send state/reward to Python
        └── MeshExporter.cs       # Export 3D mesh data
```

## Components and Their Functions

### C# Grasshopper Components

1. **SliderInfoExporter**
   - Exports information about sliders (min, max, rounding values)
   - Used by the RL environment to understand the action space

2. **ZmqListener**
   - Receives action arrays from Python via ZMQ
   - Updates Grasshopper number sliders based on received actions

3. **ZmqStateSender**
   - Sends current state and reward values to Python
   - Transmits design state for RL training

4. **MeshExporter**
   - Exports 3D mesh data from Grasshopper
   - Provides geometry for visualization in the feedback UI

### Python Modules

1. **env_simple.py**
   - Implements a Gymnasium environment for RL
   - Communicates with Grasshopper via ZMQ

2. **ppo_train.py**
   - Implementation of Proximal Policy Optimization algorithm
   - Learns optimal slider values for design objectives

3. **zmq_state_receiver.py**
   - Receives and logs state/reward data during training
   - Creates datasets for analysis

4. **analyze_integrated_data.py**
   - Analyzes training results and generates reference data
   - Clusters designs and identifies optimal solutions

5. **design_regenerator.py**
   - Recreates optimal designs for feedback collection
   - Sends actions to Grasshopper via ZMQ

### Web Interface

The feedback collection system is powered by a Flask web application that:
- Displays 3D models of generated designs
- Provides rating scales for different design aspects
- Collects textual comments and evaluations
- Manages sessions and design exploration

## Workflow

### 1. Prepare the Grasshopper Environment

1. Create your parametric design in Grasshopper
2. Add the RLHF components to your canvas:
   - SliderInfoExporter: Connect to sliders you want to control
   - ZmqListener: Set port to 5556 for receiving actions
   - ZmqStateSender: Set port to 5557 for sending state/reward
   - MeshExporter: Set port to 5558 for exporting meshes

### 2. Run Reinforcement Learning Training

```bash
# Start the state/reward receiver
python python_modules/zmq_state_receiver.py

# In a new terminal, start the PPO training
python python_modules/ppo_train.py --gh-path "path/to/your/definition.gh" --steps 10000
```

### 3. Analyze Results and Generate Reference Designs

```bash
# Analyze the collected data
python python_modules/analyze_integrated_data.py

# Regenerate top designs for feedback
python python_modules/design_regenerator.py --feedback-session 1
```

### 4. Collect Human Feedback

```bash
# Start the feedback server
cd server
python app.py
```

Then open http://localhost:5000 in your browser to access the feedback interface.

## ZMQ Port Configuration

The system uses several ZMQ ports for communication:

- **5556**: Action transmission (Python → Grasshopper)
- **5557**: State/reward transmission (Grasshopper → Python)
- **5558**: Mesh data requests (Web UI ↔ Grasshopper)

These can be configured via command-line arguments in the Python scripts.

## Troubleshooting

### Common Issues

1. **ZMQ Connection Failures**
   - Ensure all ports are available and not blocked by firewalls
   - Check that all components are properly initialized with 'Run' set to True
   - Verify the address format is correct (e.g., tcp://localhost:5556)

2. **Grasshopper Responsiveness**
   - If Grasshopper becomes unresponsive, try reducing the step rate in ppo_train.py
   - Large models may require more time between actions

3. **Mesh Export Problems**
   - Ensure your geometry is valid and properly connected to the MeshExporter
   - Check the server logs for JSON formatting errors

## Extending the System

### Adding Custom Reward Functions

Modify your Grasshopper definition to compute custom rewards based on:
- Design performance metrics (structural, environmental, etc.)
- Geometric properties (volume, surface area, etc.)
- Any other quantifiable design objectives

### Creating New Analysis Tools

The modular nature of the system allows for custom analysis workflows:
```python
# Example of adding a new analysis function
def analyze_spatial_quality(reference_data):
    # Your custom analysis code here
    return spatial_metrics
```

### Multiple Feedback Sessions

For larger studies with multiple participants:
```bash
# Create session-specific designs
python python_modules/design_regenerator.py --feedback-session 2
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This system builds on the ZMQ communication framework for Grasshopper
- PPO implementation based on Stable-Baselines3
- 3D visualization powered by Three.js
