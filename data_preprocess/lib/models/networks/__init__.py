import torch
import importlib
from lib.models.networks.discriminator import MultiscaleDiscriminator, ImageDiscriminator
from lib.models.networks.generator import ModulateGenerator
from lib.models.networks.encoder import ResSEAudioEncoder, ResNeXtEncoder, ResSESyncEncoder, FanEncoder
# import util.util as util

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'lib.models.networks.' + filename
    network = find_class_in_module(target_class_name, module_name)

    return network

def define_networks(opt, name, _type):
    net = find_network_using_name(name, _type)
    net = net(opt)
    return net


