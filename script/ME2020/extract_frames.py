import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import re
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--output_folder", 
        type=str, 
        help="Output folder"
    )
    
    parser.add_argument(
        "--video_dir", 
        type=str, 
        help="Video directory"
    )
    
    parser.add_argument(
        "--frames", 
        default=1,
        type=int,
        help="Number of frames to be extracted"
    )
    
    args = parser.parse_args()
    
    """train_video_dir = '/aloui/MediaEval/dev-set/sources/'
    test_video_dir = '/aloui/MediaEval/test-set/sources/'

    train_image_dir = 'datasets/ME/images/dc/train/'
    test_image_dir = 'datasets/ME/images/dc/test/'"""

    np.random.seed(42)

    for k, filename in enumerate(tqdm(os.listdir(args.video_dir))):
        if filename.endswith(".webm") | filename.endswith(".mp4") :
            vid_id = filename.split('.mp4')[0]
            print(vid_id)
            video_path = os.path.join(args.video_dir, filename)
            cap = cv2.VideoCapture(video_path)
            if args.frames == 1:
                frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.array([0.5]) # middle frame
                print(frameIds)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameIds[0])
                ret, frame = cap.read()
                file_name = os.path.join(args.output_folder, str(vid_id) + '.jpg')
                cv2.imwrite(file_name, frame)
            else:
                working = False 
                while not working:
                   try:
                      frameIds = (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) * np.array([(1+i) / (args.frames + 1) for i in range(args.frames)])
                      print(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) 
                      for i, fid in enumerate(frameIds):
                         cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                         ret, frame = cap.read()
                         file_name = os.path.join(args.output_folder, str(vid_id) + '_' + str(i) + '.jpg')
                         #print('shape:', frame.shape)
                         #print('size:', frame.size)
                         #print('--- arr ---\n', frame, '\n--- end ---')
                         cv2.imwrite(file_name, frame)
                      working = True
                   except:
                      print('Exception with', cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), frameIds)
                      break 
                      #print(file_name)
                      #raise Exception('Exception')

if __name__ == "__main__":

    main()

