
from . import model as model_lib


class DLBSResNet(model_lib.Model):
    """ResNet models consistent with MXNet/Caffe2 implementation that follow
       recommended 'bn-relu-conv' order.
    """
    specs = {
        'resnet18':  {'name': 'ResNet18',  'units': [2, 2, 2, 2],
                      'num_layers': 18},
        'resnet34':  {'name': 'ResNet34',  'units': [3, 4, 6, 3],
                      'num_layers': 34},
        'resnet50':  {'name': 'ResNet50',  'units': [3, 4, 6, 3],
                      'num_layers': 50},
        'resnet101': {'name': 'ResNet101', 'units': [3, 4, 23, 3],
                      'num_layers': 101},
        'resnet152': {'name': 'ResNet152', 'units': [3, 8, 36, 3],
                      'num_layers': 152},
        'resnet200': {'name': 'ResNet200', 'units': [3, 24, 36, 3],
                      'num_layers': 200},
        'resnet269': {'name': 'ResNet269', 'units': [3, 30, 48, 8],
                      'num_layers': 269}
    }

    def __init__(self, model):
        default_batch_sizes = {
            'resnet200': 32,
            'resnet269': 32
        }
        batch_size = default_batch_sizes.get(model, 32)
        super(DLBSResNet, self).__init__(DLBSResNet.specs[model]['name'], 224,
                                         batch_size, 0.005)
        self.model = model

    def add_inference(self, cnn):
        # Prepare config
        specs = DLBSResNet.specs[self.model]
        if specs['num_layers'] >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        units = specs['units']

        cnn.use_batch_norm = True
        cnn.batch_norm_config = {'decay': 0.9, 'epsilon': 2e-5, 'scale': True}
        # Conv will add batch norm and relu as well
        cnn.pad(p=3)
        # should be - pad = (3, 3)
        cnn.conv(filter_list[0], 7, 7, 2, 2, mode='VALID')
        cnn.mpool(3, 3, 2, 2)

        for i in range(num_stages):
            cnn.residual_unit(num_filter=filter_list[i+1],
                              stride=1 if i == 0 else 2,
                              dim_match=False, bottle_neck=bottle_neck)
            for _ in range(units[i] - 1):
                cnn.residual_unit(num_filter=filter_list[i+1], stride=1,
                                  dim_match=True, bottle_neck=bottle_neck)
        cnn.apool(7, 7, 1, 1)
        cnn.reshape([-1, filter_list[-1] * 1 * 1])


def create_dlbs_resnet18():
    return DLBSResNet('resnet18')


def create_dlbs_resnet34():
    return DLBSResNet('resnet34')


def create_dlbs_resnet50():
    return DLBSResNet('resnet50')


def create_dlbs_resnet101():
    return DLBSResNet('resnet101')


def create_dlbs_resnet152():
    return DLBSResNet('resnet152')


def create_dlbs_resnet200():
    return DLBSResNet('resnet200')


def create_dlbs_resnet269():
    return DLBSResNet('resnet269')
