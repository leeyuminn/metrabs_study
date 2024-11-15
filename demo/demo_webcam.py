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
    model = tfhub.load('https://bit.ly/metrabs_l')
    print("model load done.")
    c_time = time.time()
    e_time = c_time - s_time
    print("model load time: ", e_time)

    skeleton = "smpl_24"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    # video_filepath = get_video(
    #     sys.argv[1]
    # )  # You can also specify the filepath directly here.

    cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

    # 웹캠 설정
    # MJPG 형식으로 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 이미지 크기 설정 (예: 640x480 -> fhd:1920x1080(30fps) -> hd:1280*720(60fps))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 60fps로 설정


    # 웹캠이 열렸는지 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()
    
    # def get_frame_batches(video_filepath, batch_size=8):
    #     reader = imageio.get_reader(video_filepath)
    #     frames = []
    #     for frame in reader:
    #         frames.append(frame)
    #         if len(frames) == batch_size:
    #             yield np.array(frames)
    #             frames = []
    #     if frames:
    #         yield np.array(frames)

    # frame_batches = get_frame_batches(video_filepath, batch_size=8)

    # # Get the first batch to determine the shape
    # first_batch = next(frame_batches)
    # imshape = first_batch.shape[1:3]

    # # Create a new generator including the first batch
    # frame_batches = chain([first_batch], frame_batches)


    fps = cap.get(cv2.CAP_PROP_FPS)
    imshape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        viz.new_sequence_output("../YOUR_OUTPUT_VIDEO_PATH.mp4", fps=fps)

        try:
            while True:
                # 웹캠에서 한 프레임 읽기 <-batch size 1
                ret, frame = cap.read()

                # 성공적으로 프레임을 읽었는지 확인
                if ret:
                    #frame_count += 1

                    # 현재 시간
                    #current_time = time.time()

                    # 지난 시간 계산
                    #elapsed_time = current_time - start_time

                    # 지난시간이 1초가 되면
                    # if elapsed_time >= 1:
                    #     # 프레임 속도 계산 및 출력면
                    #     fps_cal = frame_count / elapsed_time
                    #     print("FPS: {:.2f}".format(fps_cal))

                    #     # 카운터 및 타이머 초기화
                    #     frame_count = 0
                    #     start_time = time.time()

                    # 프레임 표시 -> 모델 추정으로 수정할 부분!
                    frame_batch = np.expand_dims(frame, axis=0)  # Shape (1, height, width, 3)
                    pred = model.detect_poses_batched(
                        frame_batch,
                        intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                        skeleton=skeleton,
                    )

                    # Extract predictions for this single frame
                    boxes = pred["boxes"][0]
                    poses = pred["poses3d"][0]
                    viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

                    cv2.imshow('Webcam Frame', frame)

                    # 키 입력 대기 (예: 'q'를 누르면 중단)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("프레임을 읽을 수 없습니다.")
                    break
        except KeyboardInterrupt:
            # Ctrl+C를 누르면 종료
            pass




        # for frame_batch in frame_batches:
        #     pred = model.detect_poses_batched(
        #         frame_batch,
        #         intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
        #         skeleton=skeleton,
        #     )

        #     for frame, boxes, poses in zip(frame_batch, pred["boxes"], pred["poses3d"]):
        #         viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)


    # 사용이 끝났으면 웹캠을 해제
    cap.release()
    cv2.destroyAllWindows()


# def get_video(source, temppath="/tmp/video.mp4"):
#     if not source.startswith("http"):
#         return source

#     opener = urllib.request.build_opener()
#     opener.addheaders = [("User-agent", "Mozilla/5.0")]
#     urllib.request.install_opener(opener)
#     urllib.request.urlretrieve(source, temppath)
#     return temppath


if __name__ == "__main__":
    main()