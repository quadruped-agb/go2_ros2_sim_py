ARG BASE_IMG
FROM ${BASE_IMG}

# Кэширование apt-get
RUN  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y \
  ros-jazzy-desktop \
  ros-dev-tools \
  ros-jazzy-rmw-cyclonedds-cpp \
  ros-jazzy-rclcpp \
  ros-jazzy-rclpy \
  ros-jazzy-urdf \
  ros-jazzy-xacro \
  ros-jazzy-std-msgs \
  ros-jazzy-angles \
  ros-jazzy-ros2-control \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-controller-manager \
  ros-jazzy-pcl-ros \
  ros-jazzy-hardware-interface \
  ros-jazzy-pluginlib \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-ros2launch \
  ros-jazzy-slam-toolbox \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-camera-info-manager \
  ros-jazzy-rqt-robot-steering \
  ros-jazzy-tf-transformations \
  ros-jazzy-teleop-twist-keyboard \
  ros-jazzy-ros2-controllers \
  ros-jazzy-ros-gz-image \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-opennav-docking \
  ros-jazzy-apriltag-ros \
  ros-jazzy-apriltag-msgs \
  ros-jazzy-image-proc \
  ros-jazzy-image-view \
  ros-jazzy-geometry-msgs \
  ros-jazzy-nav2-simple-commander \
  ros-jazzy-tf2-ros \
  ros-jazzy-tf2 \
  ros-jazzy-velodyne-laserscan \
  tmux \
  tmuxinator \
  python3-colcon-common-extensions \
  ccache \
  ros-jazzy-topic-tools \
  python3 python3-pip

# Установка colcon-cache
RUN pip3 install --break-system-packages git+https://github.com/ruffsl/colcon-cache.git

RUN apt-get update && apt-get install -y cmake libpcl-dev && rm -rf /var/lib/apt/lists/*

# Install RGL Gazebo Plugin (Harmonic-tagged release, ubuntu24 build)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-jazzy-octomap-server \
    ros-jazzy-octomap-msgs \
    ros-jazzy-octomap-ros \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/rgl_release && cd /tmp/rgl_release && \
    curl -L -o rgl_release.zip \
      "https://github.com/RobotecAI/RGLGazeboPlugin/releases/download/v0.2.0-harmonic/RGLGazeboPlugin_ubuntu24.zip" && \
    unzip rgl_release.zip && \
    find . -name "libRobotecGPULidar.so" -exec cp {} /opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins/ \; && \
    find . -name "libRGLServerPluginInstance.so" -exec cp {} /opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins/ \; && \
    find . -name "libRGLServerPluginManager.so" -exec cp {} /opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins/ \; && \
    mkdir -p /opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins/gui && \
    find . -name "libRGLVisualize.so" -exec cp {} /opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins/gui/ \; && \
    find . -type d -name "lidar_patterns" -exec cp -r {} /rgl_patterns \; && \
    cd / && rm -rf /tmp/rgl_release
    
RUN git clone --depth 1 --branch v0.2.0-harmonic https://github.com/RobotecAI/RGLGazeboPlugin.git /tmp/rgl_repo && \
    cp -r /tmp/rgl_repo/lidar_patterns /rgl_patterns && \
    rm -rf /tmp/rgl_repo

ENV RGL_PATTERNS_DIR=/rgl_patterns

RUN pip3 install --break-system-packages ortools --ignore-installed typing_extensions

RUN pip3 install --break-system-packages 'numpy<2' --ignore-installed
RUN apt-get update && apt-get install -y \
    ros-jazzy-octomap-server \
    ros-jazzy-octomap-msgs \
    ros-jazzy-octomap-ros \
    && rm -rf /var/lib/apt/lists/*


# Копируем исходный код
COPY ../ /root/ws/src

WORKDIR /root/ws

SHELL ["/bin/bash", "-c"]

# Сборка пакетов
RUN --mount=type=cache,target=/root/.ccache \
  . /ros_entrypoint.sh && \
  colcon cache lock && \
  ccache colcon build --symlink-install \
  --mixin ccache

# Настройка entrypoint
RUN sed -i '/exec "\$@"/i umask 002' /ros_entrypoint.sh
ENV WORKSPACE_DIR=/root/ws
RUN sed -i '/exec "\$@"/i source "$WORKSPACE_DIR/install/setup.bash"' /ros_entrypoint.sh

