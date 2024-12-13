import logging
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MoConVQ')))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import transforms3d
#import poseviz

import argparse

import cameralib
import functools
from simplepyutils import FLAGS
import simplepyutils as spu
import tensorflow_inputs as tfinp

import matplotlib
matplotlib.use("TkAgg")  # Tkinter 백엔드 사용
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from MoConVQCore.Model.MoConVQ import MoConVQ


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='https://bit.ly/metrabs_s')
    parser.add_argument('--camera-id', type=int, default='1')
    parser.add_argument('--viz-downscale', type=int, default=4)
    parser.add_argument('--out-video-path', type=str, default=None)
    parser.add_argument('--out-video-fps', type=int, default=15)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--skeleton', type=str, default='smpl_24') #smpl+head_30
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--internal-batch-size', type=int, default=128)
    parser.add_argument('--max-detections', type=int, default=-1)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--detector-flip-aug', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--random', action=spu.argparse.BoolAction)
    parser.add_argument('--detector-threshold', type=float, default=0.2)
    parser.add_argument('--detector-nms-iou-threshold', type=float, default=0.7)
    parser.add_argument('--pitch', type=float, default=5)
    parser.add_argument('--camera-height', type=float, default=1000)
    spu.argparse.initialize(parser)
    logging.getLogger('absl').setLevel('ERROR')
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

def visualize_with_matplotlib(image, detections, poses3d, poses2d, edges, retracked_poses=None):
    """
    Visualizes detections, 3D poses, and 2D poses using matplotlib in real-time.
    """
    plt.clf()  # Clear the current figure for real-time updates
    
    # Subplot for the image and 2D detections
    image_ax = plt.subplot(1, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red'))

    # Subplot for the 3D pose
    pose_ax = plt.subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)
    pose_ax.set_zlim3d(-1500, 1500)

    # Adjust 3D pose coordinate system
    for pose3d, pose2d in zip(poses3d, poses2d):
        pose3d[:, [1, 2]] = pose3d[:, [2, 1]]  # Swap Y and Z axes
        pose3d[:, 2] *= -1  # Flip Z axis

        for i_start, i_end in edges:
            image_ax.plot(
                [pose2d[i_start, 0], pose2d[i_end, 0]],
                [pose2d[i_start, 1], pose2d[i_end, 1]],
                marker='o', markersize=2, color='blue')
            pose_ax.plot(
                [pose3d[i_start, 0], pose3d[i_end, 0]],
                [pose3d[i_start, 1], pose3d[i_end, 1]],
                [pose3d[i_start, 2], pose3d[i_end, 2]],
                marker='o', markersize=2, color='green')

        image_ax.scatter(pose2d[:, 0], pose2d[:, 1], s=2, color='yellow')
        pose_ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], s=2, color='red')

    # Visualize retracked poses
    if retracked_poses is not None:
        for retracked_pose in retracked_poses:
            retracked_pose[:, [1, 2]] = retracked_pose[:, [2, 1]]
            retracked_pose[:, 2] *= -1
            for i_start, i_end in edges:
                pose_ax.plot(
                    [retracked_pose[i_start, 0], retracked_pose[i_end, 0]],
                    [retracked_pose[i_start, 1], retracked_pose[i_end, 1]],
                    [retracked_pose[i_start, 2], retracked_pose[i_end, 2]],
                    marker='o', markersize=2, color='red')
            pose_ax.scatter(retracked_pose[:, 0], retracked_pose[:, 1], retracked_pose[:, 2], s=2, color='red')

    plt.draw()  # Draw the updated plot
    plt.pause(0.001)  # Pause briefly to allow the plot to update

def main():
    initialize()
    model = tfhub.load(FLAGS.model_path)
    joint_names = model.per_skeleton_joint_names[FLAGS.skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[FLAGS.skeleton].numpy()
    print("joint names: ",joint_names)
    print("joint edges", joint_edges)

    extrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix[:3, :3] = transforms3d.euler.euler2mat(0, np.deg2rad(FLAGS.pitch), 0, 'ryxz')
    extrinsic_matrix[1, 3] = FLAGS.camera_height
    camera = cameralib.Camera(
        intrinsic_matrix=np.array(
            [[616.68, 0, 301.59], [0, 618.78, 231.30], [0, 0, 1]], np.float32),
        extrinsic_matrix=extrinsic_matrix)
    predict_fn = functools.partial(
        model.detect_poses_batched, intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
        internal_batch_size=FLAGS.internal_batch_size,
        extrinsic_matrix=extrinsic_matrix[np.newaxis], detector_threshold=FLAGS.detector_threshold,
        detector_nms_iou_threshold=FLAGS.detector_nms_iou_threshold,
        detector_flip_aug=FLAGS.detector_flip_aug,
        max_detections=FLAGS.max_detections,
        antialias_factor=FLAGS.antialias_factor, num_aug=FLAGS.num_aug,
        suppress_implausible_poses=True, skeleton=FLAGS.skeleton)

    # Initialize MoConVQ model
    agent = MoConVQ(323, 12, 57, 120, env=None, training=False)
    agent.simple_load('moconvq_base.data', strict=True)
    agent.eval()

    frame_batches_gpu, frame_batches_cpu = tfinp.webcam(
        capture_id=FLAGS.camera_id, batch_size=FLAGS.batch_size, prefetch_gpu=1)

    plt.figure(figsize=(10, 5.2))  # Initialize the matplotlib figure

    frame_count = 0
    start_time = time.time()

    for frames_gpu, frames_cpu in zip(frame_batches_gpu, frame_batches_cpu):
        frames_gpu = frames_gpu[:, :, ::-1]
        frames_cpu = [f[:, ::-1] for f in frames_cpu]

        pred = predict_fn(frames_gpu)
        for frame, boxes, poses3d, poses2d in zip(
                frames_cpu, pred['boxes'].numpy(), pred['poses3d'].numpy(), pred['poses2d'].numpy()):
            print(f"3D Poses for the current frame (shape: {poses3d.shape}):")
            print(poses3d)

            # Prepare pose data for MoConVQ
            motion_array = poses3d[np.newaxis, ...]  # Add batch dimension
            latent_info = agent.encode_seq_all(motion_array, motion_array)
            retracked_poses = latent_info['latent_dynamic'][0]  # Assuming latent_dynamic corresponds to retracked poses

            # Visualize original and retracked poses
            visualize_with_matplotlib(frame, boxes, poses3d, poses2d, joint_edges, retracked_poses=retracked_poses)

            frame_count += 1
            elapsed_time = time.time() - start_time

            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

if __name__ == '__main__':
    main()
