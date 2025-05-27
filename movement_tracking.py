#%%
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bg_median_tracking(video_path):
    video = cv2.VideoCapture(video_path)

    ok, frame = video.read()

    crop = cv2.selectROI("Crop ROI", frame, False)
    cv2.destroyWindow("Crop ROI")

    fps = video.get(cv2.CAP_PROP_FPS)

    crop_x, crop_y, crop_w, crop_h = [int(v) for v in crop]

    frames = []

    while True:
        ok, frame = video.read()
        if not ok:
            break
        
        matrix = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].mean(axis=2).astype(np.uint8)
        frames.append(matrix)

    bg = np.mean(np.asarray(frames), axis=0)

    pos = []
    for frame in frames:
        dif = frame - bg

        min_index = np.unravel_index(np.argmin(dif), dif.shape)
        
        pos_x = min_index[1] + crop_x
        pos_y = min_index[0] + crop_y
        pos.append((pos_x,pos_y))
        
    pos = np.asarray(pos).T
    return *pos, fps, bg

def save_trajectory_plot(x, saving_path, filename = "x_vs_frame"):
    plt.figure()
    plt.plot(x)
    plt.xlabel("frame")
    plt.ylabel("x [px]")
    plt.savefig(os.path.join(saving_path, filename))


def save_trajectory_data(x,y,fps, saving_path, filename = "trajectory.pickle"):
    df = pd.DataFrame({"x":x,"y":y})
    df.attrs = {"fps":fps}
    pd.to_pickle(df, os.path.join(saving_path, filename))

def save_trajectory_video(x,y,fps, video_path, saving_path, filename = "tracking.mp4"):
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()

    size = (frame.shape[1] , frame.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
    out = cv2.VideoWriter(os.path.join(saving_path,"tracking.mp4"), fourcc, fps, size)

    for seed_x, seed_y in zip(x,y):
        ok, frame = video.read()
        if not ok:
            break
    
        cv2.circle(frame, (int(seed_x), int(seed_y)), 2, (0, 255, 0), 2, 1)
        out.write(frame)

def save_image(img, saving_path, filename="background"):
    plt.figure()
    plt.imshow(img)
    plt.savefig(os.path.join(saving_path, filename))

def save_movement_tracking_data(video_path, saving_path):
    x, y, fps, bg = bg_median_tracking(video_path)

    save_image(bg, saving_path)
    save_trajectory_video(x, y, fps, video_path, saving_path)    
    save_trajectory_plot(x, saving_path)
    save_trajectory_data(x, y, fps, saving_path)

def track(video_path, saving_dir, folder_name = None):
    if not folder_name:
        video_filename = os.path.basename(video_path).split(".")[0]
        folder_name = video_filename

    saving_path = os.path.join(saving_dir, folder_name)

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    save_movement_tracking_data(video_path, saving_path)

#%%
track("control_de_movimiento\DELTA45_NSTEPS40_TIMESTEP50_RUTINA_5.mp4", "saving_test", folder_name="mean")
