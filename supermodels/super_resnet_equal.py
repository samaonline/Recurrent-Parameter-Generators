import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torch.nn.modules.utils import _pair
from pdb import set_trace as st

import numpy as np

__all__ = ['ResNet', 'recurresnet18', 'superresnet18e', 'superresnet34e', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def super_conv3x3(mem, in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return SuperConv2d(mem, in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def super_conv1x1(mem, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return SuperConv2d(mem, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SuperParams(nn.Module):
    def __init__(self, n_dims, recurrent=False):
        super(SuperParams, self).__init__()
        self.n_dims = n_dims
        self.recurrent = recurrent
        self.w = torch.empty(n_dims)
        with torch.no_grad():
            # self.w.normal_(0, 0.03)
            nn.init.kaiming_normal_(self.w.view(512,-1,3,3), mode='fan_out', nonlinearity='relu')
        self.w = nn.Parameter(self.w)

        self.p_idxs = []
        self.b_masks = []
        self.nelements = []


    def add_param(self, dim_wi):
        if self.recurrent:
            self.p_idxs.append(torch.arange(dim_wi).cuda())
            self.b_masks.append(torch.ones(dim_wi).cuda())
        else:
            self.b_masks.append(nn.Parameter(torch.from_numpy(np.random.binomial(p=.5, n=1, size=dim_wi).astype(np.float32) * 2 - 1).cuda(), requires_grad = False))
            self.nelements.append(dim_wi)

        return len(self.nelements) - 1

    def finalize(self, is_training, _resume, save_name):
        if not _resume:
            # permute all params
            num_alldesparam = np.sum(np.array(self.nelements))
            lis = []
            for i in range(num_alldesparam//self.n_dims):
                lis.append( np.arange(self.n_dims) )
            lis.append( np.random.choice(self.n_dims, num_alldesparam%self.n_dims, replace=False) )
            lis = np.concatenate(lis)
            perm = torch.randperm(len(lis))
            lis = torch.from_numpy(lis).cuda()
            lis = lis[perm]
            # assign params
            start_pt = 0
            for cur_nelement in self.nelements:
                self.p_idxs.append( nn.Parameter(lis[start_pt:start_pt+cur_nelement], requires_grad = False) )
                start_pt += cur_nelement

            temp = np.array([i.cpu().numpy().astype(int) for i in self.p_idxs])
            np.save(save_name+"p_idxs", temp)
            temp = np.array([i.cpu().numpy().astype(int) for i in self.b_masks])
            np.save(save_name+"b_masks", temp)
        else:
            temp = np.load(save_name+"p_idxs.npy", allow_pickle=True)
            self.p_idxs = [nn.Parameter(torch.from_numpy(i), requires_grad = False) for i in temp]
            temp = np.load(save_name+"b_masks.npy", allow_pickle=True)
            self.b_masks = [nn.Parameter(torch.from_numpy(i), requires_grad = False) for i in temp]
    
    def forward(self, key_idx):
        perm_key = self.p_idxs[key_idx].cuda(device = self.w.device.index)
        mask_key = self.b_masks[key_idx].cuda(device = self.w.device.index)
        return self.w[perm_key]*mask_key

class SuperConv2d(nn.Module):
    def __init__(self, mem, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SuperConv2d, self).__init__()
        self.mem = mem

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # This fixes the std in the shared memory Kaiming initialization, the Kaiming initialization used 'fan_out' 'relu' mode
        self.initialization_multiplier = ((512. * 3 * 3) / (self.out_channels * kernel_size * kernel_size))**0.5


        self.lid = mem.add_param(self.in_channels*self.out_channels*self.kernel_size[0]*self.kernel_size[1])

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def assign_weight(self):
        params = self.mem(self.lid)
        params = params.view(self.out_channels,
                             self.in_channels,
                             self.kernel_size[0],
                             self.kernel_size[1])
        params = params * self.initialization_multiplier
        self.weight = nn.Parameter(params, requires_grad = False)
        #self.weight_mask = params#None
        #self.weight_orig = params#None
        
    def forward(self, x):
        if hasattr(self, "weight_mask"):
            return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        
        params = self.mem(self.lid)
        params = params.view(self.out_channels,
                             self.in_channels,
                             self.kernel_size[0],
                             self.kernel_size[1])
        params = params * self.initialization_multiplier
        return F.conv2d(x, params, self.bias, stride=self.stride, padding=self.padding)

class Myconv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, params, bias, stride=1, padding=0):
        ctx.save_for_backward(x, params, bias)
        ctx.stride = stride
        ctx.padding = padding
        return F.conv2d(x, params, bias, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_output):
        x, params, bias = ctx.saved_tensors
        grad_x = F.conv_transpose2d(grad_output, params, stride=ctx.stride, padding=ctx.padding)
        grad_params = F.conv2d(x.transpose(1,0), grad_output.transpose(1,0) , dilation=ctx.stride, padding=ctx.padding)

        import ipdb; ipdb.set_trace()

class SuperBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, mem, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(SuperBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = super_conv3x3(mem, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = super_conv3x3(mem, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SuperResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None, recurrent_params=False, mem_size = 512*512*3*3, _resume=False, save_name='temp'):
        super(SuperResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64

        self.mem = SuperParams(mem_size, recurrent=recurrent_params)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.mem.finalize(self.training, _resume = _resume, save_name = save_name)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                super_conv1x1(self.mem, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.mem, self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.mem, self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def recurresnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SuperResNet(SuperBasicBlock, [2, 2, 2, 2], recurrent_params=True, **kwargs)
    if pretrained:
        raise NotImplemented
    return model


def superresnet18e(pretrained=False, mem_size = 512*512*3*3, _resume=False, save_name="temp", **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SuperResNet(SuperBasicBlock, [2, 2, 2, 2], mem_size = mem_size, _resume=_resume, save_name=save_name, **kwargs)
    if pretrained:
        raise NotImplemented
    return model


def superresnet34e(pretrained=False, mem_size = 512*512*3*3, _resume=False, save_name="temp", **kwargs):
#def superresnet34e(pretrained=False, mem_size = 512*3*3*2422, _resume=False, save_name="temp", **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SuperResNet(SuperBasicBlock, [3, 4, 6, 3], mem_size = mem_size, _resume=_resume, save_name=save_name, **kwargs)
    if pretrained:
        raise NotImplemented
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
