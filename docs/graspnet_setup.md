# GraspNet Baseline Setup

COMPASS does not vendor GraspNet Baseline. The grasp node
`ros2_ws/src/detect_graspnet/detect_graspnet/grasp_server_node.py` expects the
GraspNet Baseline source tree to be installed locally at:

```text
~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet/graspnet-baseline
```

It imports these directories from that local checkout:

```text
graspnet-baseline/models
graspnet-baseline/dataset
graspnet-baseline/utils
```

## Create The Conda Environment

Create a dedicated environment for GraspNet:

```bash
conda create -n graspnet-test python=3.10 -y
conda activate graspnet-test
python -m pip install --upgrade pip setuptools wheel
```

Install the ROS Python packages through the ROS environment at runtime. The
COMPASS launch script sources ROS 2 before activating the conda environment.

## Install GraspNet Baseline

Install the upstream project in the package-local path:

```bash
cd ~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline

pip install -r requirements.txt
```

Compile the CUDA/C++ operators:

```bash
cd pointnet2
python setup.py install

cd ../knn
python setup.py install
```

Install GraspNet API:

```bash
cd ~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

Install COMPASS detection extras in the same environment:

```bash
pip install open3d opencv-python scipy spatialmath-python ultralytics
```

Install PyTorch for your CUDA version if it was not installed by the upstream
requirements. Use the command recommended by the PyTorch installer for your
driver/CUDA stack.

## Model Weights

`grasp_server_node.py` loads the RealSense checkpoint from:

```text
~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet/logs/log_rs/checkpoint-rs.tar
```

Create the directory and place the checkpoint there:

```bash
mkdir -p ~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet/logs/log_rs
# Put checkpoint-rs.tar in the directory above.
```

The checkpoint is a local model artifact and should not be committed to the
public repository.

## Quick Import Test

Run this after installation:

```bash
conda activate graspnet-test
cd ~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet
python - <<'PY'
import os
import sys

root = os.getcwd()
sys.path.append(os.path.join(root, "graspnet-baseline", "models"))
sys.path.append(os.path.join(root, "graspnet-baseline", "dataset"))
sys.path.append(os.path.join(root, "graspnet-baseline", "utils"))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup

print("GraspNet imports OK")
PY
```

Then test the node with ROS sourced:

```bash
source /opt/ros/humble/setup.bash
source ~/ros_workspace/COMPASS/ros2_ws/install/setup.bash
conda activate graspnet-test
cd ~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet
python3 grasp_server_node.py
```
