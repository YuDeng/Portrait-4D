#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import math
import torch
import torch.nn.init as init
import numpy as np
import lib.models as models

def make_model(cfg):
    # net object init
    facerecon = None
    fd = None
    ldmk = None
    ldmk_3d = None
    

    ############################## build model #############################################################
    return_list = []
    
    # create facerecon model
    facerecon = models.define_networks(
        cfg.model.facerecon,
        cfg.model.facerecon.model_type,
        cfg.model.facerecon.model_cls        
    )
    return_list.append(facerecon)
    
    # create fd model
    fd = models.define_networks(
        cfg, 
        cfg.model.fd.model_type,
        cfg.model.fd.model_cls
    )
    return_list.append(fd)
    
    # create ldmk model
    ldmk = models.define_networks(
        cfg, 
        cfg.model.ldmk.model_type,
        cfg.model.ldmk.model_cls
    )
    return_list.append(ldmk)

    # create ldmk 3d model
    ldmk_3d = models.define_networks(
        cfg, 
        cfg.model.ldmk_3d.model_type,
        cfg.model.ldmk_3d.model_cls
    )
    return_list.append(ldmk_3d)
    
    return return_list
