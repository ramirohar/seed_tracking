import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bg_median_tracking(video, tracking_region):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    crop_x, crop_y, crop_w, crop_h = [int(v) for v in tracking_region]

    frames = []

    while True:
        ok, frame = video.read()
        if not ok:
            break

        matrix = (
            frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            .mean(axis=2)
            .astype(np.uint8)
        )
        frames.append(matrix)

    bg = np.mean(np.asarray(frames), axis=0)

    pos = []
    for frame in frames:
        dif = frame - bg

        min_index = np.unravel_index(np.argmin(dif), dif.shape)

        pos_x = min_index[1] + crop_x
        pos_y = min_index[0] + crop_y
        pos.append((pos_x, pos_y))

    pos = np.asarray(pos).T
    return *pos, bg


def select_ROI(video):
    ok, frame = video.read()

    crop = cv2.selectROI("Crop ROI", frame, False)
    cv2.destroyWindow("Crop ROI")
    
    print("ROI selected")
    
    return crop


def save_trajectory_plot(x, saving_path, filename="x_vs_frame"):
    plt.figure()
    plt.errorbar(np.arange(len(x)),x, ls="",fmt="." , markersize=3)
    plt.xlabel("frame")
    plt.ylabel("x [px]")
    plt.savefig(os.path.join(saving_path, filename))


def save_trajectory_data(x, y, fps, saving_path, filename="trajectory.pickle"):
    df = pd.DataFrame({"x": x, "y": y})
    df.attrs = {"fps": fps}
    pd.to_pickle(df, os.path.join(saving_path, filename))


def save_trajectory_video(
    video, x, y, fps, tracking_region, saving_path, filename="tracking.mp4"
):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ok, frame = video.read()
    size = (frame.shape[1], frame.shape[0])

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID' for .avi
    out = cv2.VideoWriter(os.path.join(saving_path, "tracking.mp4"), fourcc, fps, size)

    crop_x, crop_y, crop_w, crop_h = [int(v) for v in tracking_region]

    for seed_x, seed_y in zip(x, y):
        ok, frame = video.read()
        if not ok:
            break

        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (200, 255, 255), 1)
        cv2.line(frame, (int(seed_x), crop_y) ,  (int(seed_x),crop_y - 10), (0,255,0), 3)
        cv2.line(frame, (int(seed_x),crop_y+ crop_h) ,  (int(seed_x),crop_y + crop_h + 10), (0,255,0), 3)
        
        out.write(frame)


def save_image(img, saving_path, filename="background"):
    plt.figure()
    plt.imshow(img)
    plt.savefig(os.path.join(saving_path, filename))


def save_movement_tracking_data(video_path, saving_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    tracking_region = select_ROI(video)

    x, y, bg = bg_median_tracking(video, tracking_region)

    save_image(bg, saving_path)
    save_trajectory_plot(x, saving_path)
    save_trajectory_data(x, y, fps, saving_path)
    save_trajectory_video(video, x, y, fps, tracking_region, saving_path)


def track(video_path, saving_dir, folder_name=None):
    if not folder_name:
        video_filename = os.path.basename(video_path).split(".")[0]
        folder_name = video_filename

    saving_path = os.path.join(saving_dir, folder_name)

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    save_movement_tracking_data(video_path, saving_path)

if __name__ == "__main__":
    testing = False
    while True:
        saving_dir = input("Saving directory path: ")
        if saving_dir == "testing":
            testing = True
            break
        if not os.path.isdir(saving_dir):
            print("Invalid directory path")
        else:
            print(f"Selected '{saving_dir}' as saving directory")
            break
    if not testing:
        while True:
            video_path = input("Video path: ")

            if not os.path.exists(video_path):
                print("Invalid video path, try again")
                continue
            print(f"Tracking {video_path}")
            track(video_path, saving_dir)
    else:
        track(
            "videos//control_de_movimiento//DELTA90_NSTEPS20_TIMESTEP100.mp4",
            "testing_tracking_results"
              )
