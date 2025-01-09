## Create Basic YAML

Create a folder named `cfgs/ROBOT_NAME`. Put the YAML file named `basic.yaml` describing the robot in the created folder.

The `basic.yaml` should contain:
```
robot:
  asset_path: ASSET_PATH
  scale: SCALE
  (optional)default_dof_pos:
    DOF_NAME: DEFAULT_DOF_POS
    ...

control:
  control_freq: CONTROL_FREQ
  (optional)dof_names:
    - DOF_NAME (in the order of action and observation)
  kp: KP (if all joints share the same kp)
    DOF_NAME: DOF_KP (otherwise assign kp joint by joint)
    ...
  kd: KD (if all joints share the same kd)
    DOF_NAME: DOF_KD (other wise assign kd joint by joint)
    ...
  (optional) armature: ARMATURE (if all joints share the same armature)
    DOF_NAME: ARMATURE (otherwise assign amature joint by joint)
    ...
  (optional) damping: DAMPING (if all joints share the same damping)
    DOF_NAME: DAMPING (otherwise assign damping joint by joint)
    ...

```

## Test Control Configs

Run `python control_robot.py -r ROBOT_NAME`, 

## Assign Body and Foot Links

Run `python assign_body.py -r ROBOT_NAME -n EXP_NAME`

## Adjust Initial Body Pose

Run `python adjust_body_pose.py -r ROBOT_NAME -n EXP_NAME`