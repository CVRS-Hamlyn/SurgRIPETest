# import torch
import numpy as np
from evaluate import Evaluator
import os
from natsort import natsorted
import argparse
from estimator import Estimator


def get_data_samples(root_path):
    image_paths = natsorted(os.listdir(os.path.join(root_path,'image')))
    pose_paths = natsorted(os.listdir(os.path.join(root_path,'pose')))
    image_paths = [os.path.join(root_path,'image',img_path) for img_path in image_paths]
    pose_paths = [os.path.join(root_path,'pose',pose_path) for pose_path in pose_paths]
    # mask_path = os.path.join(root_path,'mask')    #in case of someone use the masks

    return list(zip(image_paths, pose_paths))

TASK_CHOICES = {'l': 'LND', 'm': 'MBF'}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluateion script for SurgRIPE.')
    parser.add_argument('--path', help= 'Get path to data root path.')
    parser.add_argument('--type', choices=TASK_CHOICES.keys(), default='l', help= 'Instrument Type for test.')
    parser.add_argument('--spath', help= 'Get save path of results.')
    return parser.parse_args()

def main():
    args = get_args()
    instrument_type = TASK_CHOICES.get(args.type)

    estimator = Estimator()
    root_path = args.path    #the root path for dataset e.g. .../LND/TRAIN
    test_samples = get_data_samples(root_path)

    evaluator = Evaluator(root_path,instrument_type)

    for test_sample in test_samples:

        image_path, pose_path = test_sample
        
        pose_gt = np.load(pose_path)

        pose_pred = estimator.predict(image_path)

        evaluator.evaluate(pose_gt, pose_pred)

    evaluator.summarize(args.spath)


if __name__ == "__main__":
    main()

