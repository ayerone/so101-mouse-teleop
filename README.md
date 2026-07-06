# so101-mouse-teleop

[What is LeRobot?](https://github.com/ayerone/lerobot-intro)

## This Project
Using the mouse for teleoperation enables increased maneuverability (versus keyboard teleop). Moving the mouse translates the end-effector in X and Y (just like arrow keys would), and the scroll wheel adjusts the Z. Then, holding 'r' switches to rotation mode, where the gripper can be moved in "hunch"/"lift" (forward/back on the mouse) and wrist roll (mouse left/right). This lets me perform tasks like pick and place without a leader arm.

*Full controls described below*

![demo of mouse teleop](images/mouse_teleop_full.gif)

I provide a MuJoCo simulation to test-drive mouse-based teleoperation on the SO-101 robot.

**Note:** This project is *not* polished or dependable, this is very much a proof of concept/work in progress. Inverse Kinematics is a tricky game, and you should make sure you understand the risks (and you take full responsibility) before even considering running any code on physical hardware.

## Dependencies

- [MuJoCo](https://github.com/google-deepmind/mujoco) (`mujoco`)
- [placo](https://github.com/Rhoban/placo) (`placo`)
- [NumPy](https://numpy.org/) (`numpy`)

These are listed in `pyproject.toml` and I install them with:

```bash
uv sync
```

## Running

```bash
uv run python sim_mouse_teleop.py
```

This should open a MuJoCo simulation window with the SO-ARM and a tkinter GUI window (with a gray background).

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--max-joint-speed` | 60 °/s | Maximum joint speed in degrees per second, including during the home reset; this is a kind of basic safety limit, though not by any means a full safety check |
| `--show-target-frame` | off | Display XYZ axes of the "target frame" (that IK is solving for) in the MuJoCo viewer |

## Controls

*Focus must be on the gray tkinter window to capture mouse movements*

| Input | Action |
|---|---|
| Click in the tkinter window | Arm the robot |
| Move mouse | Move end effector X/Y |
| Scroll | Move end effector Z |
| Hold left button | Close gripper |
| Hold right button | Open gripper |
| Hold `r` | Orientation mode: mouse controls gripper tilt and roll |
| Hold `e` | Pause mouse input (reposition mouse without moving robot) |
| `q` | Reset to home pose and disarm |

## Robot Model Files

| File | Purpose |
|---|---|
| `model/so101_new_calib.urdf` | used by the placo kinematics solver for FK/IK |
| `model/so101_new_calib.xml` | MuJoCo MJCF robot definition, including joints, actuators, and meshes |
| `model/scene.xml` | MuJoCo scene file; includes the robot, ground plane, cube, and tray |
| `assets/` | STL mesh files for the SO-101 arm links |

[·](images/googley_eyes.md)
