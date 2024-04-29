#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

##### python internal and external package
import os
import cv2
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

##### self defined package
from lib.config.config import cfg
from lib.inferencer import Tester as Tester

def main():
    os.makedirs(cfg.save_dir, exist_ok=True)
    tester = Tester(cfg)
    tester.inference(cfg.input_dir, cfg.save_dir, video=cfg.is_video)

if __name__ == "__main__":
    main()

    
