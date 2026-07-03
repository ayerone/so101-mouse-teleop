# How to Add Googley Eyes to Your MuJoCo Robot

Each eye is two parts: a white sclera sphere fixed to the body, and a dark pupil sphere on a ball joint with near-zero stiffness. Gravity makes the pupil droop and swing as the robot moves.

Add this inside whatever body you want the eyes attached to (in this repo, the gripper jaw):

```xml
<!-- right eye -->
<geom type="sphere" class="visual" size="0.008" pos="0 -0.022 -0.004" rgba="1 1 1 0.85"/>
<body name="eye_r_pupil" pos="0 -0.022 -0.004" quat="0.930 0 0 0.367">
  <joint type="ball" name="eye_r_ball" damping="1e-6" frictionloss="0" armature="0" stiffness="0.0000175"/>
  <inertial pos="0 -0.003 0" mass="0.001" diaginertia="1.6e-8 1.6e-8 1.6e-8"/>
  <geom type="sphere" class="visual" size="0.006" pos="0 -0.003 0" rgba="0.05 0.05 0.05 1"/>
</body>

<!-- left eye -->
<geom type="sphere" class="visual" size="0.008" pos="0 -0.022  0.038" rgba="1 1 1 0.85"/>
<body name="eye_l_pupil" pos="0 -0.022  0.038" quat="0.930 0 0 0.367">
  <joint type="ball" name="eye_l_ball" damping="1e-6" frictionloss="0" armature="0" stiffness="0.0000175"/>
  <inertial pos="0 -0.003 0" mass="0.001" diaginertia="1.6e-8 1.6e-8 1.6e-8"/>
  <geom type="sphere" class="visual" size="0.006" pos="0 -0.003 0" rgba="0.05 0.05 0.05 1"/>
</body>
```

## Key parameters

- `pos` on the sclera geom and the pupil body must match -- that centers the pupil inside the sclera at rest.
- `quat` on the pupil body tilts the resting pose so the pupil droops forward rather than straight down.
- `pos="0 -0.003 0"` on the pupil geom offsets it from the ball joint center, so it hangs like a pendulum.
- `stiffness="0.0000175"` is just enough to stop the pupil from flopping wildly at slow speeds while still reacting to motion.
- `size="0.008"` for the sclera and `size="0.006"` for the pupil gives a bit of white showing around the edge.

Adjust `pos` values to place the eyes where you want them on your robot, and scale `size` to match the proportions of your link geometry.
