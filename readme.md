# Grasshopper RLHF System

Human feedback collection system for reinforcement learning in Grasshopper 3D models.

## Project Structure

```
C:\Users\valen\Desktop\Dev\6. RLHF\
├─data
│  ├─designs     # Design data storage
│  ├─feedback    # User feedback storage
│  ├─meshes      # 3D mesh data storage
│  └─zmq_logs    # ZMQ communication logs
├─python_modules
│  ├─analyze_integrated_data.py  # Analysis module
│  ├─env_simple.py               # RL environment
│  ├─ppo_train.py                # PPO training
│  └─zmq_state_receiver.py       # ZMQ state/reward receiver
└─server
   ├─static
   │  ├─css
   │  │  └─styles.css            # UI stylesheet
   │  └─js
   │     ├─design_manager.js     # Design management
   │     ├─feedback_form.js      # Feedback collection
   │     ├─main.js               # Main UI logic
   │     ├─three_viewer.js       # 3D viewer
   │     └─utils.js              # Utility functions
   ├─templates
   │  └─index.html               # Main UI template
   └─app.py                      # Flask server
```

## Prerequisites

1. **Software Requirements**
   - Rhino 6/7 with Grasshopper
   - Python 3.7 or higher
   - Visual Studio (for C# components)

2. **Python Packages**
   ```bash
   pip install flask zmq gymnasium stable-baselines3 torch werkzeug netmq numpy pandas matplotlib seaborn
   ```

3. **Directory Setup**
   ```bash
   mkdir -p data/meshes data/designs data/feedback data/zmq_logs
   mkdir -p server/static/css server/static/js server/templates
   ```

## Components Installation

### C# Grasshopper Components

1. Compile the C# components in Visual Studio:
   - SliderInfoExporter.cs
   - ZmqListener.cs
   - ZmqStateSender.cs
   - MeshExporter.cs

2. Convert the compiled DLL to GHA format and place in Grasshopper Components folder:
   - Typical path: `%APPDATA%\Grasshopper\Libraries`

## Execution Sequence

### 1. Prepare Grasshopper Environment

1. Open Rhino and load your Grasshopper file (e.g., `C:/Users/valen/Desktop/Dev/AS_B.gh`)
2. Add the following components to your canvas:
   - SliderInfoExporter: Exports slider information
   - ZmqListener: Receives actions from Python (5556 port)
   - ZmqStateSender: Sends state/reward data to Python (5557 port)
   - MeshExporter: Exports 3D meshes (5558 port)
3. Configure components:
   - ZmqListener: Set `Run` to `True`, Address to `tcp://localhost:5556`
   - ZmqStateSender: Set `Run` to `True`, Address to `tcp://localhost:5557`
   - MeshExporter: Set `Run` to `True`, Address to `tcp://localhost:5558`

### 2. Run Reinforcement Learning Pipeline

#### 2.1. Start State/Reward Receiver

```bash
cd C:\Users\valen\Desktop\Dev\6. RLHF
python python_modules\zmq_state_receiver.py
```

This script listens for state and reward data from Grasshopper on port 5557 and saves it to JSON files in the `data/zmq_logs` directory.

#### 2.2. Run PPO Training

```bash
cd C:\Users\valen\Desktop\Dev\6. RLHF
python python_modules\ppo_train.py --gh-path "C:/Users/valen/Desktop/Dev/AS_B.gh" --compute-url "http://localhost:6500/grasshopper" --port 5556 --steps 1000
```

Parameters:
- `--gh-path`: Path to your Grasshopper definition file
- `--compute-url`: URL of Rhino.Compute server
- `--port`: ZMQ port for sending actions (should match ZmqListener port)
- `--steps`: Number of training steps

#### 2.3. Analyze Results

```bash
cd C:\Users\valen\Desktop\Dev\6. RLHF
python python_modules\analyze_integrated_data.py --state-reward-log "data\zmq_logs\state_reward_log_YYYYMMDD_HHMMSS.json"
```

Replace `YYYYMMDD_HHMMSS` with the actual timestamp of the log file created by zmq_state_receiver.py. This script analyzes the state/reward data and generates reference data for the human feedback system.

### 3. Human Feedback Collection

#### 3.1. Start Flask Server

```bash
cd C:\Users\valen\Desktop\Dev\6. RLHF\server
python app.py
```

The Flask server will start on port 5000 and provide the web interface for human feedback collection.

#### 3.2. Access Web Interface

Open a web browser and navigate to:
```
http://localhost:5000
```

#### 3.3. Provide Feedback

1. Select a design from the list on the left
2. Explore the 3D model using the viewer
3. Rate the design using the sliders on the right
4. Add optional comments
5. Submit your feedback
6. Continue with the next design

## Data Flow

1. **RL Training Data Flow**:
   - Grasshopper (ZmqListener) ← Actions ← PPO Training
   - Grasshopper (ZmqStateSender) → State/Reward → ZmqStateReceiver

2. **Analysis Data Flow**:
   - ZmqStateReceiver → Log Files → Analysis Module
   - Analysis Module → Reference Data → Feedback System

3. **Feedback Collection Data Flow**:
   - Grasshopper (MeshExporter) → Mesh Data → Flask Server
   - Flask Server → Web Interface → Human User
   - Human User → Feedback → Flask Server → Feedback Files

## Port Usage

- **5556**: PPO actions to Grasshopper (PUSH/PULL)
- **5557**: State/reward data from Grasshopper (PUSH/PULL)
- **5558**: Mesh data from Grasshopper (REQ/REP)
- **5000**: Flask web server (HTTP)

## Troubleshooting

### 1. ZMQ Connection Issues

If ZMQ connections fail:
- Ensure all ports are available (not used by other applications)
- Check that Grasshopper components have `Run` set to `True`
- Verify ZMQ address format (`tcp://localhost:PORT`)
- Restart Rhino/Grasshopper

### 2. Rhino.Compute Issues

If Rhino.Compute fails to connect:
- Ensure Rhino.Compute server is running
- Check that the URL is correct
- Verify Grasshopper file path

### 3. Mesh Export Issues

If 3D meshes fail to load:
- Check MeshExporter connections in Grasshopper
- Ensure the mesh exists and is valid
- Inspect Flask server logs for errors

### 4. Flask Server Issues

If Flask server fails to start:
- Check for port conflicts (default: 5000)
- Verify required packages are installed
- Check Python version compatibility

## Customizing for Different Grasshopper Files

To use the system with different Grasshopper files:

1. **Adjust Parameter Names**:
   - Modify `env_simple.py` to match your Grasshopper component parameter names
   - Ensure `state_output_param_name` and `reward_output_param_name` match your design

2. **Update Mesh Connections**:
   - Connect MeshExporter to your 3D mesh objects

3. **Configure Training**:
   - Update the `--gh-path` parameter when running `ppo_train.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Grasshopper and Rhino3D by Robert McNeel & Associates
- Stable-Baselines3 for reinforcement learning algorithms
- Three.js for 3D visualization