import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
# from util import util

# model_urls = {
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
# }
def copy_state_dict(state_dict, model, strip=None, replace=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and replace is None and name.startswith(strip):
            name = name[len(strip):]
        if strip is not None and replace is not None:
            name = name.replace(strip, replace)
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

class ResNeXt50(nn.Module):
    def __init__(self, opt):
        super(ResNeXt50, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
        self.opt = opt
        # self.reduced_id_dim = opt.reduced_id_dim
        self.conv1x1 = nn.Conv2d(512 * Bottleneck.expansion, 512, kernel_size=1, padding=0)
        self.fc = nn.Linear(512 * Bottleneck.expansion, opt.data.num_classes)
        # self.fc_pre = nn.Sequential(nn.Linear(512 * Bottleneck.expansion, self.reduced_id_dim), nn.ReLU())


    def load_pretrain(self, load_path):
        check_point = torch.load(load_path)
        copy_state_dict(check_point, self.model)

    def forward_feature(self, input):
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        net = self.model.avgpool(x)
        net = torch.flatten(net, 1)
        x = self.conv1x1(x)
        # x = self.fc_pre(x)
        return net, x

    def forward(self, input):
        input_batch = input.view(-1, self.opt.model.output_nc, self.opt.data.img_size, self.opt.data.img_size)
        net, x = self.forward_feature(input_batch)
        net = net.view(-1, self.opt.num_inputs, 512 * Bottleneck.expansion)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(-1, self.opt.num_inputs, 512, 7, 7)
        net = torch.mean(net, 1)
        x = torch.mean(x, 1)
        cls_scores = self.fc(net)

        return [net, x], cls_scores 
        # net is feature with dim all from channel; 
        # x is feature with dim all from channel, but one more conv added and another 7*7 spatial size
