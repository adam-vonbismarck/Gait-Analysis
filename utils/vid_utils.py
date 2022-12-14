import cv2 as cv
import numpy as np
from utils.img_utils import process_frames
from utils.preprocess import preprocess

def video_to_frames(video_filepath):
    video = cv.VideoCapture(video_filepath)
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    scene_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    scene_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))

    frames = np.zeros((int(total_frames / 3), scene_height, scene_width))
    i = 0
    was_read = True

    while (i < total_frames - 1):
        was_read, frame = video.read()
        if not was_read:
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if ((i % 3) == 0):
            frames[int(i/3)] = gray_frame
        i += 1
    return frames

def get_imgs_from_vid_path(train_paths, val_paths):
    image_array = np.zeros((train_paths.shape[0]), dtype=object)
    val_array = np.zeros((val_paths.shape[0]), dtype=object)

    for i in range(train_paths.shape[0]):
        frames = video_to_frames(train_paths[i])
        imgs = process_frames(frames)
        data_sample = np.zeros((imgs.shape[0], 210, 70))
        # Import images
        for j, img in enumerate(imgs):
            # Treat edge case where the image is just black
            if np.sum(img == 255) == 0:
                continue
            img = preprocess(img)
            data_sample[j] = img
            
        image_array[i] = data_sample

    for i in range(val_paths.shape[0]):

        frames = video_to_frames(val_paths[i])
        imgs = process_frames(frames)
        data_sample = np.zeros((imgs.shape[0], 210, 70))
        # Import images
        for j, img in enumerate(imgs):
            # Treat edge case where the image is just black
            if np.sum(img == 255) == 0:
                continue
            img = preprocess(img)
            data_sample[j] = img

        val_array[i] = data_sample

    return image_array, val_array