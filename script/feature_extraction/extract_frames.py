import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import re
import numpy as np
import argparse
import pandas as pd 


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
    
    parser.add_argument(
        "--frame_list", 
        default=0,
        type=str,
        help="Files with index of frames to be extracted per video"
    )
    
    
    
    args = parser.parse_args()
    
    """train_video_dir = '/aloui/MediaEval/dev-set/sources/'
    test_video_dir = '/aloui/MediaEval/test-set/sources/'

    train_image_dir = 'datasets/ME/images/dc/train/'
    test_image_dir = 'datasets/ME/images/dc/test/'"""

    np.random.seed(42)



    for k, filename in enumerate(tqdm(os.listdir(args.video_dir))):
        if filename.endswith(".mp4") :
            vid_id = filename.split('.mp4')[0]
        elif filename.endswith(".webm"):
            vid_id = filename.split('.webm')[0]
        else :
            continue
            
        video_path = os.path.join(args.video_dir, filename)
        cap = cv2.VideoCapture(video_path)
        if args.frame_list!=0: 
            args.frames=0            
        if args.frames == 1:
            frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.array([0.5]) # middle frame
            print(frameIds)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIds[0])
            ret, frame = cap.read()
            file_name = os.path.join(args.output_folder, str(vid_id) + '.jpg')
            print(vid_id)
            cv2.imwrite(file_name, frame)

        elif args.frames == 0:
            
            working = False 
            while not working:
               try:
                  df=pd.read_csv(args.frame_list)
                  print(vid_id)
                  frameIds=df['frame'][df['video']==int(vid_id)].tolist()
                  #print(frameIds)
                  print('HEEEEERE')
                  for i, fid in enumerate(frameIds):
                     cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                     ret, frame = cap.read()
                     file_name = os.path.join(args.output_folder, str(vid_id) + '_' + str(fid) + '.jpg')
                     #print('shape:', frame.shape)
                     #print('size:', frame.size)
                     #print('--- arr ---\n', frame, '\n--- end ---')
                     cv2.imwrite(file_name, frame)
                  working = True
               except:
                  print(frame)
                  print('Exception with', cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), frameIds)
                  raise Exception('Exception')


        else:
            working = False 
            while not working:
               try:
                  print('THEEEEERE')
                  frameIds = (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) * np.array([(1.0+i) / (args.frames + 1) for i in range(args.frames)])
                  print(frameIds)
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
                  raise Exception('Exception')




if __name__ == "__main__":

    main()
