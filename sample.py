import torch
import numpy as np
from evaluate import Evaluator

model = None

test_samples = []

def image_reader(img_path):


    return None


evaluator = Evaluator()


for test_sample in test_samples:
    image_path, pose_path = test_sample

    image = image_reader(image_path)
    
    pose_gt = np.load(pose_path)

    pose_pred = model(image)

    evaluator.evaluate(pose_gt, pose_pred)

evaluator.summarize()

