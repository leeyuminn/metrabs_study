import sys
import time
import urllib.request

import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_io as tfio

import cameralib
import poseviz
import imageio
import numpy as np
from itertools import chain


def main():
    print("model load start.")
    s_time = time.time()
    model = tfhub.load('https://bit.ly/metrabs_l')
    print("model load done.")
    c_time = time.time()
    e_time = c_time - s_time
    print("model load time: ", e_time)

    skeleton = "smpl_24"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    video_filepath = get_video(
        sys.argv[1]
    )  # You can also specify the filepath directly here.
    
    
    def get_frame_batches(video_filepath, batch_size=8):
        reader = imageio.get_reader(video_filepath)
        frames = []
        for frame in reader:
            frames.append(frame)
            if len(frames) == batch_size:
                yield np.array(frames)
                frames = []
        if frames:
            yield np.array(frames)

    frame_batches = get_frame_batches(video_filepath, batch_size=8)

    # Get the first batch to determine the shape
    first_batch = next(frame_batches)
    imshape = first_batch.shape[1:3]

    # Create a new generator including the first batch
    frame_batches = chain([first_batch], frame_batches)


    camera = cameralib.Camera.from_fov(
        fov_degrees=55, imshape=first_batch.shape[1:3]
    )

    with imageio.get_reader(video_filepath) as reader:
        fps = reader.get_meta_data()["fps"]
    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        viz.new_sequence_output("../YOUR_OUTPUT_VIDEO_PATH.mp4", fps=fps)
        for frame_batch in frame_batches:
            pred = model.detect_poses_batched(
                frame_batch,
                intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                skeleton=skeleton,
            )

            for frame, boxes, poses in zip(frame_batch, pred["boxes"], pred["poses3d"]):
                viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)


def get_video(source, temppath="/tmp/video.mp4"):
    if not source.startswith("http"):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath


if __name__ == "__main__":
    main()