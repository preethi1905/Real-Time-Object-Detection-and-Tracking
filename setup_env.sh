#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Script for Project 1: Real-Time Object Detection & Tracking
# Target OS: Ubuntu 22.04 LTS (Native or Windows WSL2)
# ROS Distribution: ROS 2 Humble Hawksbill
# ==============================================================================

set -e

echo "=========================================================="
echo " Starting Setup for ROS 2 Humble & Tracking Dependencies  "
echo "=========================================================="

# 1. Update package lists
sudo apt update && sudo apt install -y curl gnupg lsb-release build-essential

# 2. Check if ROS 2 Humble is installed
if ! dpkg -s ros-humble-desktop >/dev/null 2>&1; then
    echo "--> Adding ROS 2 repository..."
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    echo "--> Installing ROS 2 Humble Desktop..."
    sudo apt install -y ros-humble-desktop
else
    echo "--> ROS 2 Humble is already installed."
fi

# 3. Install ROS 2 packages & build tools
echo "--> Installing ROS 2 dependencies and build tools..."
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-rqt-image-view \
    ros-humble-image-transport \
    ros-humble-message-filters \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    libeigen3-dev

# 4. Initialize rosdep if not already initialized
if [ ! -d "/etc/ros/rosdep" ]; then
    echo "--> Initializing rosdep..."
    sudo rosdep init || true
    rosdep update || true
fi

# 5. Install Python dependencies
echo "--> Installing Python dependencies from requirements.txt..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 6. Source ROS 2 setup in bashrc if not already present
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "--> Appending ROS 2 source to ~/.bashrc..."
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi

echo "=========================================================="
echo " Setup Complete! "
echo " Remember to run: source ~/.bashrc"
echo " To build the ROS 2 workspace:"
echo "   colcon build --symlink-install"
echo "   source install/setup.bash"
echo "=========================================================="
