import argparse
import genesis as gs

gs.init(backend=gs.cpu)

parser = argparse.ArgumentParser()
parser.add_argument('--asset', type=str, default='urdf/go2/urdf/go2.urdf')
parser.add_argument('--scale', type=float, default=1.0)

args = parser.parse_args()

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        gravity=(0, 0, 0),
    ),  
)

asset_file = args.asset
scale = args.scale


if asset_file.endswith(".urdf"):
    morph = gs.morphs.URDF(file=asset_file, collision=True, scale=scale)
elif asset_file.endswith(".xml"):
    morph = gs.morphs.MJCF(file=asset_file, collision=True, scale=scale)
else:
    raise ValueError(f"Unsupported file format: {asset_file}")
robot = scene.add_entity(
    morph,
    # surface=gs.surfaces.Default(
    #     vis_mode="visual",
    # ),
)

scene.build()


link_names = [link.name for link in robot.links]
joint_names = [joint.name for joint in robot.joints]

print(f'Links : {link_names}')
print(f'Joints: {joint_names}')

while True:
    scene.step()
