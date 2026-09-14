"""Render Grip 0 only for the 3x3 material grid (E x sigma_y) from the shared settled state.

Same settled state, grip-0 start pose, 40x12 actions, camera (notebook update_camera, cam 1) and spp for all materials.
Video frames: settled initial state + 40 env.step frames (30 close + 10 retreat).

Run: python render_material_grid.py --settled-state <pkl> --action-seq <npz>
"""
import argparse
import os
import pickle
import subprocess
from datetime import datetime

import cv2
import numpy as np

import gen_multigrip_debug as g
import material_sensitivity_debug as msd

E_VALUES = [2500, 5000, 15000]        # grid rows
YIELD_VALUES = [50, 800, 3200]        # grid columns
SPP = 3                               # notebook render_multi(spp=3)
FPS = 10


def update_camera(env):
    # copied from notebook cell 3
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)
    env.render_cfg.defrost()
    env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.2)
    env.render_cfg.camera_rot_1 = (0.8, 0.)
    env.render_cfg.camera_pos_2 = (2.4, 2.5, 0.2)
    env.render_cfg.camera_rot_2 = (0.8, 1.8)
    env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0.2)
    env.render_cfg.camera_rot_3 = (0.8, -1.8)
    env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
    env.render_cfg.camera_rot_4 = (0.8, 3.14)


def render(env):
    return env.render_multi(mode='rgb_array', spp=SPP)[0][0]   # rgb list, camera 1 (notebook rgb_0)


def encode(frame_dir, mp4):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(frame_dir, '%03d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settled-state', required=True)
    ap.add_argument('--action-seq', required=True)
    args = ap.parse_args()

    with open(args.settled_state, 'rb') as f:
        settled = pickle.load(f)
    seq = dict(np.load(args.action_seq))
    pose, actions = seq['grip_start_tool_pose'][0], seq['primitive_action_sequence'][0]
    assert actions.shape == (g.N_FRAMES, 12)

    out_dir = os.path.join(g.HERE, '..', '..', 'dataset', 'material_grid_video_' + datetime.now().strftime("%d-%b-%Y-%H:%M:%S"))
    env, _ = g.make_env()
    update_camera(env)
    env.number_of_cams = 1   # instance attribute; render_multi loops only over camera 1

    frames = {}
    for E in E_VALUES:
        for sy in YIELD_VALUES:
            name = f'E{E}_sy{sy}'
            print(name)
            msd.restore(env, settled, dict(E=E, yield_stress=sy))
            assert all(v == 0 for v in msd.state_error(env.get_state(), settled).values())
            imgs = [render(env)]                              # settled initial state (before teleport)
            g.update_primitive(env, pose[0], pose[1])         # grip 0 start pose
            g.select_tool(env, msd.TOOL_SIZE)
            assert np.array_equal(g.read_tool_pose(env), pose)
            for act in actions:                               # 30 close + 10 retreat
                env.step(act)
                imgs.append(render(env))
            frame_dir = os.path.join(out_dir, name, 'frames')
            os.makedirs(frame_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(frame_dir, f'{i:03d}.png'), img[..., ::-1])
            encode(frame_dir, os.path.join(out_dir, f'{name}.mp4'))
            frames[name] = imgs

    grid_dir = os.path.join(out_dir, 'grid_frames')
    os.makedirs(grid_dir, exist_ok=True)
    for i in range(len(actions) + 1):
        rows = []
        for E in E_VALUES:
            tiles = []
            for sy in YIELD_VALUES:
                tile = np.ascontiguousarray(frames[f'E{E}_sy{sy}'][i][..., ::-1])
                for color, thick in [((0, 0, 0), 5), ((255, 255, 255), 2)]:   # outlined label
                    cv2.putText(tile, f'E={E}  sigma_y={sy}', (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)
                tiles.append(tile)
            rows.append(np.hstack(tiles))
        cv2.imwrite(os.path.join(grid_dir, f'{i:03d}.png'), np.vstack(rows))
    encode(grid_dir, os.path.join(out_dir, 'grid_3x3_grip0.mp4'))
    print(f'outputs: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
