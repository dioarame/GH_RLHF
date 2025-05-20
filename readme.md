# Grasshopper RLHF System

A comprehensive framework for Reinforcement Learning with Human Feedback (RLHF) in Grasshopper 3D modeling environments.

## Overview

This system enables the application of reinforcement learning techniques to parametric architectural designs created in Grasshopper, with a human feedback collection mechanism that helps refine the RL model based on human preferences and design expertise.

The framework consists of:
1. **Reinforcement Learning components** (Python) that interact with Grasshopper models
2. **ZMQ communication layer** (C# + Python) for bidirectional data exchange
3. **Human feedback interface** (Flask web application) for collecting design evaluations
4. **Analysis tools** (Python) for processing and integrating feedback data
5. **Reward model** training system that incorporates human preferences

## System Architecture

```
┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
│  Grasshopper    │◄──ZMQ────►│  Python RL      │◄────────►│  Human          │
│  Environment    │           │  Environment    │          │  Feedback UI    │
└─────────────────┘           └─────────────────┘          └─────────────────┘
         ▲                            ▲                           ▲
         │                            │                           │
         │                            │                           │
         └────────────────────────────┴───────────────────────────┘
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
# Core dependencies
pip install flask zmq gymnasium stable-baselines3 torch werkzeug numpy pandas matplotlib seaborn

# Additional dependencies for analysis
pip install scikit-learn
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
│   ├── models/      # Trained reinforcement learning models
│   ├── processed_feedback/  # Processed feedback for reward models
│   └── zmq_logs/    # Communication logs for debugging
│
├── python_modules/
│   ├── analyze_integrated_data.py  # Analysis tools for RLHF data
│   ├── design_regenerator.py       # Utility to regenerate optimal designs
│   ├── enhanced_env.py             # Enhanced RL environment with human feedback
│   ├── feedback_processor.py       # Process human feedback for reward models
│   ├── reward_adapter.py           # Adapter for different reward functions
│   ├── reward_fn_optimized.py      # Optimized architecture reward function
│   ├── reward_fn_original.py       # Original enhanced reward function
│   ├── reward_function.py          # Basic seasonal reward function
│   └── rl_architecture_optimizer.py # Main RL architecture optimization system
│
├── server/
│   ├── app.py                   # Flask server for feedback collection
│   ├── static/                  # Web UI static assets (JS, CSS)
│   │   ├── js/
│   │   │   ├── main.js              # Main application logic
│   │   │   ├── three_viewer.js      # Three.js 3D viewer 
│   │   │   ├── design_manager.js    # Design list management
│   │   │   ├── utils.js             # Utility functions
│   │   │   └── feedback_form.js     # Feedback form handling
│   │   └── css/
│   │       └── styles.css           # Custom stylesheets
│   └── templates/               # HTML templates for feedback UI
│       └── index.html           # Main page template
│
└── src/                         # C# source code
    ├── GrasshopperZmqComponent.sln # Visual Studio solution
    └── GrasshopperZmqComponent/
        ├── SliderInfoExporter.cs   # Export slider ranges
        ├── ZmqListener.cs          # Receive actions from Python
        ├── ZmqStateSender.cs       # Send state/reward to Python
        └── MeshExporter.cs         # Export 3D mesh data
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

1. **rl_architecture_optimizer.py**
   - Main implementation of the architecture optimization system
   - Integrates RL environment, ZMQ communication, and reward functions

2. **reward_adapter.py**
   - Provides a unified interface to different reward functions
   - Supports switching between original, enhanced, and optimized reward functions

3. **reward_fn_optimized.py**
   - Optimized reward function based on analysis data
   - Considers BCR, FAR, seasonal sunlight, and surface-to-volume ratio

4. **enhanced_env.py**
   - Enhanced Gymnasium environment that integrates human feedback
   - Combines environmental and human-based rewards

5. **feedback_processor.py**
   - Processes human feedback data into preference pairs
   - Prepares training data for reward models

6. **analyze_integrated_data.py**
   - Analyzes training results and generates reference data
   - Clusters designs and identifies optimal solutions

7. **design_regenerator.py**
   - Recreates optimal designs for feedback collection
   - Sends actions to Grasshopper via ZMQ

### Web Interface (Server)

The feedback collection system is powered by a Flask web application that:
- Displays 3D models of generated designs using Three.js
- Provides rating scales for different design aspects (aesthetics, functionality, innovation, feasibility)
- Collects textual comments and evaluations
- Manages design exploration and feedback sessions

## RLHF Workflow

### 1. Prepare the Grasshopper Environment

1. Create your parametric design in Grasshopper
2. Add the RLHF components to your canvas:
   - SliderInfoExporter: Connect to sliders you want to control
   - ZmqListener: Set port to 5556 for receiving actions
   - ZmqStateSender: Set port to 5557 for sending state/reward
   - MeshExporter: Set port to 5558 for exporting meshes

### 2. Run Reinforcement Learning Training

```bash
# Start the training with basic PPO algorithm
python python_modules/rl_architecture_optimizer.py --steps 10000 --bcr-limit 70.0 --far-min 200.0 --far-max 500.0 --use-seasonal-reward

# Options:
# --steps: Number of training steps
# --bcr-limit: BCR legal limit (percent)
# --far-min: Minimum FAR legal limit (percent)
# --far-max: Maximum FAR legal limit (percent)
# --use-seasonal-reward: Use seasonal sunlight reward function
# --reward-type: Choose between "original", "enhanced", or "optimized" reward function
# --port: ZMQ port for action transmission (default: 5556)
# --state-port: ZMQ port for state reception (default: 5557)
```

### 3. Analyze Results and Generate Reference Designs

```bash
# Analyze the collected data
python python_modules/analyze_integrated_data.py

# Regenerate top designs for feedback collection
python python_modules/design_regenerator.py --feedback-session 1
```

### 4. Collect Human Feedback

```bash
# Start the feedback server
cd server
python app.py
```

Then open http://localhost:5000 in your browser to access the feedback interface.

### 5. Process Feedback and Train Reward Model

```bash
# Process collected feedback into preference pairs
python python_modules/feedback_processor.py

# Train reward model (implementation not included)
# This would use the preference pairs to train a neural network
```

### 6. Use Enhanced Environment with Reward Model

```bash
# Use the enhanced environment with human feedback reward model
python python_modules/enhanced_env.py --gh-path "path/to/your/definition.gh" --reward-model "path/to/reward_model.pt"
```

## Reward Functions

The system includes multiple reward functions with different characteristics:

1. **Original Reward Function** (`reward_function.py`)
   - Basic seasonal reward function
   - Considers BCR, FAR, summer and winter sunlight

2. **Enhanced Reward Function** (`reward_fn_original.py`)
   - Improved stability and training dynamics
   - Gaussian distribution-based scoring
   - Reward smoothing for better learning

3. **Optimized Reward Function** (`reward_fn_optimized.py`)
   - Based on analysis of optimal designs
   - Target ranges derived from data
   - Includes surface-to-volume ratio
   - Fine-tuned for architectural quality

## ZMQ Communication

The system uses several ZMQ ports for communication:

- **5556**: Action transmission (Python → Grasshopper)
- **5557**: State/reward transmission (Grasshopper → Python)
- **5558**: Mesh data requests (Web UI ↔ Grasshopper)

These can be configured via command-line arguments in the Python scripts.

## Architectural Optimization Parameters

The system optimizes several key architectural parameters:

- **BCR (Building Coverage Ratio)**: Percentage of the site covered by building
- **FAR (Floor Area Ratio)**: Ratio of total floor area to site area
- **Winter Sunlight**: Sunlight exposure during winter (higher is better)
- **Surface-to-Volume Ratio**: Ratio of building surface to volume (lower is better for energy efficiency)

## Troubleshooting

### Common Issues

1. **ZMQ Connection Failures**
   - Ensure all ports are available and not blocked by firewalls
   - Check that all components are properly initialized with 'Run' set to True
   - Verify the address format is correct (e.g., tcp://localhost:5556)

2. **Grasshopper Responsiveness**
   - If Grasshopper becomes unresponsive, try reducing the step rate
   - Large models may require more time between actions

3. **Mesh Export Problems**
   - Ensure your geometry is valid and properly connected to the MeshExporter
   - Check the server logs for JSON formatting errors

4. **Invalid Designs**
   - The system includes automatic retry mechanisms for invalid geometries
   - Ensure your Grasshopper definition produces closed breps

## Extending the System

### Adding Custom Reward Functions

Create a new reward function by implementing:
1. A class with a `calculate_reward(state)` method
2. State normalization and scoring logic
3. Register it in the `reward_adapter.py` file

```python
# Example:
class CustomRewardFunction:
    def __init__(self, params):
        # Initialize parameters
        
    def calculate_reward(self, state):
        # Calculate reward from state
        return reward, info
        
    def reset_prev_state(self):
        # Reset internal state
```

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
