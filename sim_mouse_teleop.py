#!/usr/bin/env python
"""
Mouse end-effector teleoperation of the SO-101 in MuJoCo simulation.

Run from the repo root:
  uv run python sim_mouse_teleop.py
  uv run python sim_mouse_teleop.py --max-joint-speed 30

Controls:
  Left-click tkinter window         Arm the robot
  Move mouse                        X/Y EE velocity (delta per frame)
  Scroll                            Z up/down (cumulative)
  Hold left button (after arming)   Close gripper
  Hold right button (after arming)  Open gripper
  Hold r                            Orientation mode: mouse controls gripper spherical angles
                                    (theta = mouse Y, phi = mouse X, both from where r was pressed)
                                    Release r to return to position mode.
  Hold e                            Pause mouse (reposition cursor without moving robot)
  q                                 Reset simulation to home pose and disarm
"""

import argparse
import enum
import platform
import time
import tkinter as tk
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import placo

# ── Inlined utilities ─────────────────────────────────────────────────────────

def precise_sleep(seconds: float, spin_threshold: float = 0.010, sleep_margin: float = 0.005) -> None:
    if seconds <= 0:
        return
    if platform.system() in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            if remaining > spin_threshold:
                time.sleep(max(remaining - sleep_margin, 0))
    else:
        time.sleep(seconds)


class RobotKinematics:
    def __init__(self, urdf_path: str, target_frame_name: str, joint_names: list[str]) -> None:
        self.robot  = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.joint_names      = joint_names
        self.target_frame_name = target_frame_name
        self.tip_frame        = self.solver.add_frame_task(target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        for name, val in zip(self.joint_names, np.deg2rad(joint_pos_deg[:len(self.joint_names)])):
            self.robot.set_joint(name, val)
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        for name, val in zip(self.joint_names, np.deg2rad(current_joint_pos[:len(self.joint_names)])):
            self.robot.set_joint(name, val)
        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)
        self.solver.solve(True)
        self.robot.update_kinematics()
        joint_pos_deg = np.rad2deg([self.robot.get_joint(n) for n in self.joint_names])
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[:len(self.joint_names)]  = joint_pos_deg
            result[len(self.joint_names):]  = current_joint_pos[len(self.joint_names):]
            return result
        return joint_pos_deg


# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent
MJCF_PATH    = str(_HERE / "model" / "scene.xml")
URDF_PATH    = str(_HERE / "model" / "so101_new_calib.urdf")
TARGET_FRAME = "gripper_frame_link"

# ── Configuration ─────────────────────────────────────────────────────────────

FPS           = 30
GRIPPER_SPEED = 5.0   # degrees per frame

EE_BOUNDS_MIN = np.array([-0.50, -0.50,  0.00])
EE_BOUNDS_MAX = np.array([ 0.50,  0.50,  0.60])

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
HOME_Q_DEG  = np.array([0.0, -60.0, 60.0, 30.0, 0.0, 50.0])

MOUSE_SENSITIVITY    = 0.001   # meters per pixel (full window ~= 0.4 m of travel)
SCROLL_STEP          = 0.01    # meters per scroll tick
Z_SMOOTH             = 0.25    # fraction of remaining Z gap closed per frame (~0.3 s settle)
MAX_EE_STEP          = 0.01    # meters per frame -- caps how far the IK target can jump

ORIENT_SENSITIVITY   = 0.005   # radians per pixel (100 px ≈ 29°)

POS_ORIENT_WEIGHT    = 0.01    # orientation weight during position mode (IK default)
ORIENT_POS_WEIGHT    = 1.0     # position weight during orientation mode
ORIENT_ORIENT_WEIGHT = 1.0     # orientation weight during orientation mode

DEFAULT_MAX_JOINT_SPEED = 60.0  # degrees per second

_INSTRUCTIONS = (
    "mouse        →  forward, back, left, right\n"
    "scroll       →  target height\n"
    "hold r       →  orient (θ/φ)\n"
    "hold e       →  pause mouse (to re-center)\n"
    "left btn     →  close gripper\n"
    "right btn    →  open gripper\n"
    "q            →  reset"
)

# Wayland doesn't expose window position; suppress the repeated GLFW complaint.
warnings.filterwarnings("ignore", message=".*Wayland.*window position.*")

# ── State machine ─────────────────────────────────────────────────────────────

class AppState(enum.Enum):
    WAITING   = "waiting"
    TRACKING  = "tracking"
    RESETTING = "resetting"

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SimEnv:
    model: mujoco.MjModel
    data:  mujoco.MjData
    qpos_ids: list[int]
    ctrl_ids: list[int]
    cube_body_id:       int
    gripper_body_id:    int
    moving_jaw_body_id: int
    cube_qpos_adr:  int
    cube_dof_adr:   int
    cube_init_qpos: np.ndarray
    cube_geoms:      frozenset[int]
    fixed_jaw_geoms: frozenset[int]
    moving_jaw_geoms: frozenset[int]


@dataclass
class MouseState:
    # cursor
    state:    AppState       = field(default=AppState.WAITING)
    pos:      tuple | None   = None
    last_pos: tuple | None   = None
    # scroll / Z
    scroll:   int   = 0
    z_target: float = 0.0
    z_offset: float = 0.0
    # buttons
    gripper_close: bool = False
    gripper_open:  bool = False
    mouse_pause:   bool = False
    # orientation mode (r held)
    orient_mode:       bool             = False
    orient_origin:     tuple | None     = None
    T_at_r_press:      np.ndarray | None = None
    orient_axis_1:     np.ndarray | None = None
    orient_axis_2:     np.ndarray | None = None
    r_release_pending: bool             = False


@dataclass
class GraspState:
    active: bool             = False
    offset: np.ndarray | None = None   # cube origin in gripper body frame
    qrel:   np.ndarray | None = None   # cube quat relative to gripper quat

    def acquire(
        self,
        cube_pos: np.ndarray, cube_quat: np.ndarray,
        grip_pos: np.ndarray, grip_rot: np.ndarray, grip_quat: np.ndarray,
    ) -> None:
        q2inv = grip_quat * np.array([1., -1., -1., -1.])
        self.offset = grip_rot.T @ (cube_pos - grip_pos)
        self.qrel   = quat_mul(q2inv, cube_quat)
        self.active = True

    def release(self) -> None:
        self.active = False
        self.offset = None
        self.qrel   = None


@dataclass
class ControlState:
    q_target:    np.ndarray   # joint angles in degrees (all DOF incl. gripper)
    T_desired:   np.ndarray   # 4×4 target EE transform
    gripper_pos: float        # gripper target in degrees


@dataclass
class UiVars:
    status:       tk.StringVar
    instructions: tk.StringVar
    orient:       tk.StringVar
    ee:           tk.StringVar

# ── Helpers ───────────────────────────────────────────────────────────────────

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix. axis need not be normalized, angle in radians."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def collision_geom_ids(model, body_id: int) -> frozenset[int]:
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    return frozenset(i for i in range(start, start + count) if model.geom_contype[i] > 0)


def geoms_in_contact(data, set_a: frozenset[int], set_b: frozenset[int]) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in set_a and c.geom2 in set_b) or \
           (c.geom2 in set_a and c.geom1 in set_b):
            return True
    return False


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two MuJoCo quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def apply_scroll_z(mouse: MouseState) -> float:
    """Consume pending scroll ticks, advance the smoothed Z offset, return the delta."""
    mouse.z_target += mouse.scroll * SCROLL_STEP
    mouse.scroll = 0
    prev = mouse.z_offset
    mouse.z_offset += (mouse.z_target - mouse.z_offset) * Z_SMOOTH
    return mouse.z_offset - prev


def draw_frame_axes(user_scn, pos: np.ndarray, rot_mat: np.ndarray, length: float = 0.05) -> None:
    """Draw X/Y/Z axes (red/green/blue) at pos using rot_mat columns as axis directions."""
    rgba = [[1., .2, .2, 1.], [.2, 1., .2, 1.], [.2, .2, 1., 1.]]
    for i in range(3):
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        z = rot_mat[:, i]
        z = z / np.linalg.norm(z)
        tmp = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
        x = np.cross(tmp, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mujoco.mjv_initGeom(
            user_scn.geoms[user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([0.003, 0.006, length]),
            pos,
            np.column_stack([x, y, z]).flatten(),
            np.array(rgba[i], dtype=np.float32),
        )
        user_scn.ngeom += 1

# ── Setup ─────────────────────────────────────────────────────────────────────

def load_sim() -> SimEnv:
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)

    qpos_ids = [model.joint(name).qposadr.item() for name in MOTOR_NAMES]
    ctrl_ids = [int(model.actuator(name).id)      for name in MOTOR_NAMES]

    cube_body_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "cube")
    gripper_body_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "gripper")
    moving_jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "moving_jaw_so101_v1")
    cube_jnt_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_adr      = int(model.jnt_qposadr[cube_jnt_id])
    cube_dof_adr       = int(model.jnt_dofadr[cube_jnt_id])

    return SimEnv(
        model=model, data=data,
        qpos_ids=qpos_ids, ctrl_ids=ctrl_ids,
        cube_body_id=cube_body_id,
        gripper_body_id=gripper_body_id,
        moving_jaw_body_id=moving_jaw_body_id,
        cube_qpos_adr=cube_qpos_adr,
        cube_dof_adr=cube_dof_adr,
        cube_init_qpos=model.qpos0[cube_qpos_adr:cube_qpos_adr + 7].copy(),
        cube_geoms=collision_geom_ids(model, cube_body_id),
        fixed_jaw_geoms=collision_geom_ids(model, gripper_body_id),
        moving_jaw_geoms=collision_geom_ids(model, moving_jaw_body_id),
    )


def init_home(env: SimEnv, kin: RobotKinematics) -> ControlState:
    for i, (qpos_id, ctrl_id) in enumerate(zip(env.qpos_ids, env.ctrl_ids)):
        env.data.qpos[qpos_id] = np.radians(HOME_Q_DEG[i])
        env.data.ctrl[ctrl_id] = np.radians(HOME_Q_DEG[i])
    mujoco.mj_forward(env.model, env.data)
    q0 = np.degrees(np.array([env.data.qpos[i] for i in env.qpos_ids]))
    return ControlState(
        q_target=q0.copy(),
        T_desired=kin.forward_kinematics(q0),
        gripper_pos=float(q0[MOTOR_NAMES.index("gripper")]),
    )


def build_ui() -> tuple[tk.Tk, UiVars]:
    root = tk.Tk()
    root.title("Mouse Teleop")
    root.geometry("525x425")

    ui = UiVars(
        status       =tk.StringVar(value="Click in the center to start"),
        instructions =tk.StringVar(value=_INSTRUCTIONS),
        orient       =tk.StringVar(value="θ:   +0.0°   φ:   +0.0°"),
        ee           =tk.StringVar(value=""),
    )

    tk.Label(root, textvariable=ui.status,       font=("Mono", 12)).pack(pady=10)
    tk.Label(root, textvariable=ui.orient,       font=("Mono", 10)).pack()
    tk.Label(root, textvariable=ui.ee,           font=("Mono", 10)).pack()
    tk.Label(root, textvariable=ui.instructions, font=("Mono", 10), justify=tk.LEFT).pack(pady=10)

    return root, ui


def bind_events(
    root:  tk.Tk,
    mouse: MouseState,
    grasp: GraspState,
    ctrl:  ControlState,
    env:   SimEnv,
    kin:   RobotKinematics,
    ui:    UiVars,
) -> None:

    def on_left_press(e):
        if mouse.state == AppState.WAITING:
            mouse.last_pos = (e.x_root, e.y_root)
            mouse.pos      = (e.x_root, e.y_root)
            mouse.z_target = 0.0
            mouse.z_offset = 0.0
            mouse.state    = AppState.TRACKING
            ui.status.set("press q to reset")
            ui.instructions.set("")
        else:
            mouse.gripper_close = True

    def on_left_release(e):
        mouse.gripper_close = False

    def on_right_press(e):
        if mouse.state == AppState.TRACKING:
            mouse.gripper_open = True

    def on_right_release(e):
        mouse.gripper_open = False

    def on_motion(e):
        mouse.pos = (e.x_root, e.y_root)

    def on_scroll(e):
        mouse.scroll += 1 if e.num == 4 else -1

    def on_scroll_wheel(e):
        mouse.scroll += 1 if e.delta > 0 else -1

    def on_r_press(e):
        mouse.r_release_pending = False
        if mouse.state != AppState.TRACKING or mouse.pos is None:
            return
        if mouse.orient_mode:
            return  # already active (key repeat)
        mouse.orient_mode   = True
        mouse.orient_origin = mouse.pos
        mouse.T_at_r_press  = ctrl.T_desired.copy()
        # axis_1: horizontal, normal to the vertical plane through robot base and gripper tip
        gx, gy = ctrl.T_desired[0, 3], ctrl.T_desired[1, 3]
        h = np.array([-gy, gx, 0.0])
        n = np.linalg.norm(h)
        mouse.orient_axis_1 = h / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        mouse.orient_axis_2 = ctrl.T_desired[:3, 2].copy()

    def _commit_r_release():
        if not mouse.r_release_pending:
            return
        mouse.r_release_pending = False
        mouse.orient_mode       = False
        if mouse.pos is not None and mouse.state == AppState.TRACKING:
            mouse.last_pos = mouse.pos

    def on_r_release(e):
        mouse.r_release_pending = True
        root.after(50, _commit_r_release)  # 50 ms debounce filters key-repeat releases

    def on_q_press(e):
        if mouse.state == AppState.RESETTING:
            return
        mouse.z_target          = 0.0
        mouse.z_offset          = 0.0
        mouse.scroll            = 0
        mouse.orient_mode       = False
        mouse.r_release_pending = False
        mouse.gripper_close     = False
        mouse.gripper_open      = False
        if mouse.pos is not None:
            mouse.last_pos = mouse.pos
        grasp.release()
        env.data.qpos[env.cube_qpos_adr:env.cube_qpos_adr + 7] = env.cube_init_qpos
        env.data.qvel[env.cube_dof_adr:env.cube_dof_adr + 6]   = 0.
        mouse.state = AppState.RESETTING
        ui.status.set("returning to home...")
        ui.instructions.set("")

    def on_e_press(e):
        mouse.mouse_pause = True

    def on_e_release(e):
        mouse.mouse_pause = False
        if mouse.pos is not None:
            mouse.last_pos = mouse.pos

    root.bind("<Button-1>",        on_left_press)
    root.bind("<ButtonRelease-1>", on_left_release)
    root.bind("<Button-3>",        on_right_press)
    root.bind("<ButtonRelease-3>", on_right_release)
    root.bind("<Motion>",          on_motion)
    root.bind("<Button-4>",        on_scroll)
    root.bind("<Button-5>",        on_scroll)
    root.bind("<MouseWheel>",      on_scroll_wheel)
    root.bind("<KeyPress-r>",      on_r_press)
    root.bind("<KeyRelease-r>",    on_r_release)
    root.bind("<KeyPress-q>",      on_q_press)
    root.bind("<KeyPress-e>",      on_e_press)
    root.bind("<KeyRelease-e>",    on_e_release)

# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(
    env:   SimEnv,
    kin:   RobotKinematics,
    mouse: MouseState,
    grasp: GraspState,
    ctrl:  ControlState,
    ui:    UiVars,
    root:  tk.Tk,
    max_joint_delta: float,
    show_gripper_frame: bool = False,
) -> None:
    n_substeps = max(1, round(1.0 / (FPS * env.model.opt.timestep)))

    with mujoco.viewer.launch_passive(
        env.model, env.data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            try:
                root.update()
            except tk.TclError:
                break

            if mouse.state == AppState.RESETTING:
                gripper_idx = MOTOR_NAMES.index("gripper")
                arm_diff    = HOME_Q_DEG[:gripper_idx] - ctrl.q_target[:gripper_idx]
                ctrl.q_target[:gripper_idx] += np.clip(arm_diff, -max_joint_delta, max_joint_delta)
                grip_diff    = HOME_Q_DEG[gripper_idx] - ctrl.gripper_pos
                ctrl.gripper_pos += np.clip(grip_diff, -max_joint_delta, max_joint_delta)
                ctrl.T_desired[:] = kin.forward_kinematics(ctrl.q_target)
                if np.max(np.abs(arm_diff)) <= max_joint_delta and abs(grip_diff) <= max_joint_delta:
                    ctrl.q_target[:] = HOME_Q_DEG
                    ctrl.gripper_pos = float(HOME_Q_DEG[gripper_idx])
                    ctrl.T_desired[:] = kin.forward_kinematics(ctrl.q_target)
                    mouse.state = AppState.WAITING
                    ui.status.set("Click in the center to start")
                    ui.instructions.set(_INSTRUCTIONS)

            elif mouse.state == AppState.TRACKING and mouse.pos is not None:

                if mouse.orient_mode and mouse.orient_origin is not None:
                    # ── Orientation mode: mouse → spherical angles of gripper ──
                    dx_px = mouse.pos[0] - mouse.orient_origin[0]
                    dy_px = mouse.pos[1] - mouse.orient_origin[1]

                    # theta: tilt (rotate around axis_1 — horizontal, ⊥ to approach direction)
                    # phi:   azimuth (rotate around world Z — vertical)
                    dtheta = -dy_px * ORIENT_SENSITIVITY  # mouse up → tilt gripper up
                    dphi   =  dx_px * ORIENT_SENSITIVITY  # mouse right → swing gripper right

                    z_delta = apply_scroll_z(mouse)
                    mouse.T_at_r_press[2, 3] = np.clip(
                        mouse.T_at_r_press[2, 3] + z_delta, EE_BOUNDS_MIN[2], EE_BOUNDS_MAX[2],
                    )

                    R_theta = rotation_matrix(mouse.orient_axis_1, dtheta)
                    R_phi   = rotation_matrix(mouse.orient_axis_2, dphi)
                    ctrl.T_desired[:3, :3] = R_phi @ R_theta @ mouse.T_at_r_press[:3, :3]
                    ctrl.T_desired[:3, 3]  = mouse.T_at_r_press[:3, 3]  # XY locked, Z from scroll

                    q_raw = kin.inverse_kinematics(
                        ctrl.q_target, ctrl.T_desired,
                        position_weight=ORIENT_POS_WEIGHT,
                        orientation_weight=ORIENT_ORIENT_WEIGHT,
                    )
                    ctrl.q_target = np.clip(
                        q_raw, ctrl.q_target - max_joint_delta, ctrl.q_target + max_joint_delta,
                    )

                else:
                    # ── Position mode: mouse delta → XY EE position ───────────
                    if mouse.mouse_pause:
                        dx_px, dy_px = 0, 0
                    else:
                        dx_px = mouse.pos[0] - mouse.last_pos[0]
                        dy_px = mouse.pos[1] - mouse.last_pos[1]
                    mouse.last_pos = mouse.pos

                    z_delta = apply_scroll_z(mouse)

                    # Screen axes → robot axes:
                    #   mouse right (+dx) → robot Y-  (arm moves right)
                    #   mouse down  (+dy) → robot X-  (arm pulls back)
                    #   scroll up         → robot Z+  (arm rises)
                    ee_delta = np.array([
                        -dy_px * MOUSE_SENSITIVITY,
                        -dx_px * MOUSE_SENSITIVITY,
                        z_delta,
                    ])
                    step = np.linalg.norm(ee_delta[:2])
                    if step > MAX_EE_STEP:
                        ee_delta[:2] *= MAX_EE_STEP / step
                    ctrl.T_desired[:3, 3] = np.clip(
                        ctrl.T_desired[:3, 3] + ee_delta, EE_BOUNDS_MIN, EE_BOUNDS_MAX,
                    )

                    q_raw = kin.inverse_kinematics(
                        ctrl.q_target, ctrl.T_desired,
                        orientation_weight=POS_ORIENT_WEIGHT,
                    )
                    ctrl.q_target = np.clip(
                        q_raw, ctrl.q_target - max_joint_delta, ctrl.q_target + max_joint_delta,
                    )
                    ctrl.T_desired[:3, :3] = kin.forward_kinematics(ctrl.q_target)[:3, :3]

            # ── Gripper ───────────────────────────────────────────────────────
            if mouse.state == AppState.TRACKING and not grasp.active:
                if mouse.gripper_close:
                    ctrl.gripper_pos = max(0.0,   ctrl.gripper_pos - GRIPPER_SPEED)
                elif mouse.gripper_open:
                    ctrl.gripper_pos = min(100.0, ctrl.gripper_pos + GRIPPER_SPEED)

            # ── Grasp / release ───────────────────────────────────────────────
            cube_pos = env.data.xpos[env.cube_body_id].copy()
            if not grasp.active and mouse.gripper_close and \
                    geoms_in_contact(env.data, env.moving_jaw_geoms, env.cube_geoms) and \
                    geoms_in_contact(env.data, env.fixed_jaw_geoms,  env.cube_geoms):
                grasp.acquire(
                    cube_pos,
                    env.data.xquat[env.cube_body_id].copy(),
                    env.data.xpos[env.gripper_body_id].copy(),
                    env.data.xmat[env.gripper_body_id].reshape(3, 3).copy(),
                    env.data.xquat[env.gripper_body_id].copy(),
                )
            elif grasp.active and mouse.gripper_open:
                grasp.release()

            # ── HUD ───────────────────────────────────────────────────────────
            jaw = ctrl.T_desired[:3, 2]
            theta_disp = np.degrees(np.arcsin(np.clip(jaw[2], -1.0, 1.0)))
            phi_disp   = ctrl.q_target[MOTOR_NAMES.index("wrist_roll")]
            ui.orient.set(f"θ: {theta_disp:+6.1f}°   φ: {phi_disp:+6.1f}°")
            x, y, z = ctrl.T_desired[:3, 3]
            ui.ee.set(f"EE  x={x:+.3f}  y={y:+.3f}  z={z:+.3f} m")

            # ── Step simulation ───────────────────────────────────────────────
            for i, name in enumerate(MOTOR_NAMES):
                val = ctrl.gripper_pos if name == "gripper" else ctrl.q_target[i]
                env.data.ctrl[env.ctrl_ids[i]] = np.radians(val)

            for _ in range(n_substeps):
                mujoco.mj_step(env.model, env.data)

            if grasp.active:
                p2 = env.data.xpos[env.gripper_body_id].copy()
                R2 = env.data.xmat[env.gripper_body_id].reshape(3, 3).copy()
                q2 = env.data.xquat[env.gripper_body_id].copy()
                env.data.qpos[env.cube_qpos_adr:env.cube_qpos_adr + 3] = p2 + R2 @ grasp.offset
                env.data.qpos[env.cube_qpos_adr + 3:env.cube_qpos_adr + 7] = quat_mul(q2, grasp.qrel)
                env.data.qvel[env.cube_dof_adr:env.cube_dof_adr + 6] = 0.
                mujoco.mj_forward(env.model, env.data)

            viewer.user_scn.ngeom = 0
            if show_gripper_frame:
                draw_frame_axes(viewer.user_scn, ctrl.T_desired[:3, 3], ctrl.T_desired[:3, :3])
            viewer.sync()

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

        viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Mouse EE teleop in MuJoCo simulation")
    parser.add_argument(
        "--max-joint-speed", type=float, default=DEFAULT_MAX_JOINT_SPEED,
        metavar="DEG_S",
        help=f"Maximum joint speed in degrees/second (default: {DEFAULT_MAX_JOINT_SPEED})",
    )
    parser.add_argument(
        "--show-target-frame", action="store_true", default=False,
        help="Overlay XYZ axes on the IK target frame",
    )
    args = parser.parse_args()

    kin   = RobotKinematics(urdf_path=URDF_PATH, target_frame_name=TARGET_FRAME, joint_names=MOTOR_NAMES)
    env   = load_sim()
    ctrl  = init_home(env, kin)
    root, ui = build_ui()
    mouse = MouseState()
    grasp = GraspState()
    bind_events(root, mouse, grasp, ctrl, env, kin, ui)
    run_loop(env, kin, mouse, grasp, ctrl, ui, root, max_joint_delta=args.max_joint_speed / FPS, show_gripper_frame=args.show_target_frame)
    print()


if __name__ == "__main__":
    main()
