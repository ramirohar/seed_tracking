#%%
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
def bg_mean_tracking(video_path):
    video = cv2.VideoCapture(video_path)

    ok, frame = video.read()

    crop = cv2.selectROI("Parte a cropear", frame, False)
    cv2.destroyWindow("Parte a cropear")

    crop_x, crop_y, crop_w, crop_h = [int(v) for v in crop]

    frames = []

    while True:
        ok, frame = video.read()
        if not ok:
            break
        
        matrix = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].mean(axis=2).astype(np.uint8)
        frames.append(matrix)

    bg = np.asarray(frames).mean(axis=0)

    pos = []
    for i,frame in enumerate(frames):
        dif = frame - bg

        min_index = np.unravel_index(np.argmin(dif), dif.shape)
        
        pos_x = min_index[1] + crop_x
        pos_y = min_index[0] + crop_y
        pos.append((pos_x,pos_y))
        
    pos = np.asarray(pos).T
    return pos

#%%

def save_tracking_data(video_path, save_path):
    x,y = bg_mean_tracking(video_path)

    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()

    fps = video.get(cv2.CAP_PROP_FPS)
    size = (frame.shape[1] , frame.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
    out = cv2.VideoWriter(os.path.join(save_path,"tracking_practice.mp4"), fourcc, fps, size)

    print(frame)
    for seed_x,seed_y in zip(x,y):
        ok, frame = video.read()
        if not ok:
            break
    
        cv2.circle(frame, (int(seed_x), int(seed_y)), 2, (0, 255, 0), 2, 1)
        out.write(frame)


    plt.plot(x)
    plt.savefig(os.path.join(save_path, "x_vs_frame"))

    df = pd.DataFrame({"x":x,"y":y})
    df.attrs = {"fps":fps}
    pd.to_pickle(df, os.path.join(save_path, "dataframe.pickle"))

#%%
save_tracking_data("control_de_movimiento\DELTA45_NSTEPS40_TIMESTEP50_RUTINA_5.mp4  ", "saving_test")

