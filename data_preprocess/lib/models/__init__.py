#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

##### python internal and external package
import importlib
##### self defined package
from lib.models.fd.fd import faceDetector
from lib.models.ldmk.ldmk import ldmkDetector


def find_class_in_module(target_cls_name, module):
    # target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if target_cls_name == name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def find_network_using_name(target_class_name, filename):
    module_name = 'lib.models.{}.{}'.format(filename, filename)
    network = find_class_in_module(target_class_name, module_name)

    return network

def define_networks(opt, _type, _cls):
    net = find_network_using_name(_cls, _type)
    net = net(opt)
    return net