# System-level setup (ROS 2 Humble · Gazebo Classic 11 · Docker)

These require `sudo` and are host-level (not in the venv). Verified target: **Ubuntu
22.04 Jammy, x86_64** — the officially supported platform for ROS 2 Humble + Gazebo
Classic 11. Needed from **Week 3** (Gazebo evaluation); the 2D-sim work in Weeks 1–2
does not need any of this.

Run each block in your terminal, or in this session by prefixing with `!`.

---

## 1. ROS 2 Humble (desktop) + TurtleBot3

```bash
# 1a. Locale
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# 1b. Enable universe + add the ROS 2 apt repo
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 1c. Install ROS 2 Humble desktop (includes rviz2) + dev tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-humble-desktop ros-dev-tools

# 1d. TurtleBot3 + Gazebo Classic 11 packages (Classic ships as gazebo11 on Humble)
sudo apt install -y \
    ros-humble-turtlebot3 ros-humble-turtlebot3-msgs ros-humble-turtlebot3-simulations \
    ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# 1e. Shell setup (append to ~/.bashrc)
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export TURTLEBOT3_MODEL=waffle' >> ~/.bashrc
# turtlebot3_gazebo ships NO GAZEBO_MODEL_PATH hook -> set it or the robot renders invisibly:
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models' >> ~/.bashrc
source ~/.bashrc

# 1f. Verify
gazebo --version                     # expect 11.x
ros2 pkg list | grep turtlebot3      # expect several packages
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo empty_world.launch.py   # a Waffle should spawn
```

---

## 2. Docker (for the training-image parity with Kaggle)

```bash
# Official convenience script
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sudo sh /tmp/get-docker.sh

# Run docker without sudo (log out/in afterwards for the group to take effect)
sudo usermod -aG docker "$USER"

# Verify
docker --version
docker run --rm hello-world      # after re-login
```

Build the training image (from repo root):

```bash
docker build -t can-nav:train .
docker run --rm can-nav:train    # runs the smoke tests inside the container
```

---

## 3. HuNavSim (Week 3, 3-day timebox — plan D7)

Attempt the HuNavSim Gazebo-Classic wrapper for behavior-realistic pedestrians. If the
Classic-11 wrapper fights the integration (known risk), fall back to Gazebo actor plugins
driven by our own SFM node — budget max 3 days before invoking the fallback.

```bash
# (Week 3) clone into the ROS 2 workspace and build with colcon; details TBD at Week 3.
```

## Notes

- Disk was ~31 GB free at scaffold time; ROS 2 desktop + Gazebo + Docker ≈ 6–8 GB.
- If `gazebo --version` reports Classic 11, you are good. Do **not** install Ignition/gz
  (the plan targets Gazebo **Classic** 11).

## Troubleshooting

### Gazebo GUI (`gzclient`) does not open, but `gzserver` runs and the robot spawns

Symptom: `ros2 launch turtlebot3_gazebo empty_world.launch.py` logs
`Successfully spawned entity [waffle]` (physics is running) but **no GUI window appears**.
Running `gzclient --verbose` shows:

```
[Err] [GuiIface.cc:124] inotify_add_watch(...) failed: (No space left on device)
```

This is **not** a disk-space problem. `inotify_add_watch` returning `ENOSPC` means the
kernel inotify **watch/instance limit** is exhausted (VS Code, codex, Brave, file-sync
tools eat them). Gazebo's Qt GUI needs inotify file-watchers, so the window fails to open.

Fix (raise the limits):

```bash
# immediate (until reboot)
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=1024
# persistent
echo -e "fs.inotify.max_user_watches=524288\nfs.inotify.max_user_instances=1024" \
    | sudo tee /etc/sysctl.d/60-inotify.conf
sudo sysctl -p /etc/sysctl.d/60-inotify.conf
```

Defaults on this machine were `max_user_watches=65536`, `max_user_instances=128`.

### Gazebo GUI opens but the TurtleBot3 is invisible (spawns but not rendered)

Symptom: the launch reports `Successfully spawned entity [waffle]` and the empty world
renders (ground + sun), but the robot itself is not visible.

Cause: `gzclient` resolves the robot's visual meshes via `model://turtlebot3_common/...`,
which requires `GAZEBO_MODEL_PATH` to include the TB3 models directory. But sourcing
`/opt/ros/humble/setup.bash` leaves `GAZEBO_MODEL_PATH` **empty** — the `turtlebot3_gazebo`
package ships `path`/`ament_prefix_path` env hooks but **no `GAZEBO_MODEL_PATH` hook**.
So the meshes never load and the robot renders as nothing.

Fix (add to `~/.bashrc`, after `export TURTLEBOT3_MODEL=waffle`):

```bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
```

Then `source ~/.bashrc` and relaunch. The Waffle will be visible.
This should be part of the step 1e shell setup.
