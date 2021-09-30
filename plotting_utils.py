import tensorflow as tf
from time import process_time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def generate_images(model, test_input, tar, batch_num, output_location):
    """ uses a trained generator and test input to generate ouputs
    """
    test_input = tf.expand_dims(test_input, axis=0)
    tar = tf.expand_dims(tar, axis=0)
    start_t = process_time()
    print('Generating IR scene {0}'.format(batch_num))
    prediction = model(test_input, training=False)
    output_num = 0
    for vis, ir, pred in zip(test_input, tar, prediction):
        # each tensor contains 64 frames of footage
        for idx in range(64):
            display_list = [vis[idx], ir[idx], pred[idx]]
            title = ['Input', 'Ground Truth', 
                    'Output']

            fig, axs = plt.subplots(1, 3, figsize=(20,10))
            for i in range(3):
                axs[i].imshow(display_list[i] * .5 + .5)
                axs[i].set_title(title[i])
                axs[i].axis('off')

            # plt.show()
            plt.savefig(output_location + '/output_{0}_{1}'.format(batch_num, str(output_num).zfill(7)))
            plt.close()
            output_num += 1

def render_video(output_dir='./output'):
    """ turns images in the output directory into a video.
    """
    frame_paths = sorted([output_dir + '/' + i for i in os.listdir(output_dir)])
    frame_array = list()
    for frame in frame_paths:
        # read image
        img = cv2.imread(frame)
        height, width, channels = np.shape(img)
        size = (width, height)
        frame_array.append(img)

    # name the video after its segment
    out_file = output_dir + '/eval_vid.mp4'
    print('writing ', len(frame_array), ' frames to ', out_file, '...\n')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vidWriter = cv2.VideoWriter(out_file, fourcc, 30.0, size)

    # write video
    for image in frame_array:
        vidWriter.write(image)
    vidWriter.release()