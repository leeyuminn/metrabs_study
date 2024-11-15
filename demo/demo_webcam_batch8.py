import sys
import time
import cv2
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

    frame_count = 0

    print("model load start.")
    s_time = time.time()
    model = tfhub.load('https://bit.ly/metrabs_s')
    print("model load done.")
    c_time = time.time()
    e_time = c_time - s_time
    print("model load time: ", e_time)

    skeleton = "smpl_24"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 60fps 설정

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("cap.get fps: {:.2f}".format(fps))
    imshape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        # viz.new_sequence_output("../YOUR_OUTPUT_VIDEO_PATH.mp4", fps=fps)

        start_time = time.time()
        frame_count = 0
        batch_size = 8
        frames = []  # Batch 프레임 저장 리스트

        try:
            while True:
                ret, frame = cap.read()

                if ret:
                    frames.append(frame)  # 프레임 저장
                    frame_count += 1

                    # Batch 크기 조건 충족 시 추론 수행
                    if len(frames) == batch_size:
                        frame_batch = np.array(frames)  # Batch 배열 생성

                        # 모델 추론
                        pred = model.detect_poses_batched(
                            frame_batch,
                            intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                            skeleton=skeleton,
                        )

                        # Batch 결과 처리 및 시각화
                        for frame, boxes, poses in zip(frames, pred["boxes"], pred["poses3d"]):
                            viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

                        # Batch 초기화
                        frames = []

                    # FPS 계산
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    if elapsed_time >= 1:
                        fps_cal = frame_count / elapsed_time
                        print("FPS: {:.2f}".format(fps_cal))

                        # 카운터 및 타이머 초기화
                        frame_count = 0
                        start_time = time.time()

                else:
                    print("프레임을 읽을 수 없습니다.")
                    break

        except KeyboardInterrupt:
            print("프로그램이 종료되었습니다.")

        finally:
            cap.release()
            # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
