#!/usr/bin/env python
# coding: utf-8

import glob
import os
import numpy as np
import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--output_folder", type=str, 
        default="datasets/ME/features_100/dc/ME_test_resnext152_faster_rcnn_genome_average.lmdb", 
        help="Output folder"
    )
    parser.add_argument(
        "--features_dir",
        type=str, 
        default="datasets/ME/features_100/dc/ME_test_resnext152_faster_rcnn_genome.lmdb", 
        help="Features directory, e.g., datasets/ME/features_100/ME_trainval_resnext152_faster_rcnn_genome.lmdb"
    )
    
    args = parser.parse_args()
    
    #features_dir = "datasets/ME/features_100/dc/ME_trainval_resnext152_faster_rcnn_genome.lmdb"

    infiles = glob.glob(os.path.join(args.features_dir, "*"))

    # check and remove files with errors
    print("Cheking and removing files with errors")
    count_ = {}
    trun = []
    for infile in tqdm.tqdm(infiles):
         reader = np.load(infile, allow_pickle=True)
         fid = reader.item().get("image_id")
         #print(fid)
         #id_ = int(fid.split('_')[0])
         id_=fid[:-2]
         #print(id_)
         s= fid[-1]
         #print(s)
         #s = int(fid.split('_')[1])
         if id_ in count_:
          count_[id_] += 1
         else:
          count_[id_] = 1
        #except as e :
            #print(str(e))
            #trun.append(os.path.splitext(os.path.basename(infile))[0])
            #os.remove(infile)

    print("{} errors were found.".format(len(trun)))
    print("{} video features available.".format(len(count_)))
    
    feats = {}

    for infile in tqdm.tqdm(infiles):
        reader = np.load(infile, allow_pickle=True)
        fid = reader.item().get("image_id")
        id_ = fid[:-2]
        s = fid[-1]
        if id_ in feats:
            feats[id_]["image_id"] = str(id_)
            # perform an average of the existing feature and the new one
            feats[id_]["features"] = np.mean( np.array([ feats[id_]["features"], reader.item().get("features") ]), axis=0 )
        else:
            # initialize the sample
            feats[id_] = reader.item()

    # features_dir = "datasets/ME/features_100/dc/ME_trainval_resnext152_faster_rcnn_genome.lmdb/average"

    for img_id, feat in tqdm.tqdm(feats.items()):
        file_name = str(img_id) + ".npy"
        np.save(os.path.join(args.output_folder, file_name), feat)


if __name__ == "__main__":

    main()
