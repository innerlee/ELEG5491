#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-all

import os, sys
parrots_home = os.environ.get('PARROTS_HOME')
if not parrots_home:
    raise EnvironmentError("The environment variable 'PARROTS_HOME' must be set.")
sys.path.append(os.path.join(parrots_home, 'parrots', 'python'))
from pyparrots import dnn
from pyparrots.dnn import layerprotos
from pyparrots.dnn.modules import ModuleProto, GModule
from pyparrots.dnn.layerprotos import Convolution, GlobalPooling, Pooling, Concat, Sum
from pyparrots.dnn.layerprotos import BN, ReLU, Dropout, SoftmaxWithLoss, Accuracy
from pyparrots.base import set_debug_log
set_debug_log(True)

def FullyConnected(out, init="gauss(0.01)", decay=1):
    fc = layerprotos.FullyConnected(out)
    fc.param_policies[0] = {"init": init, "decay_mult": decay}
    fc.param_policies[1] = {"init": "fill(0)"}
    return fc

def Convolution(ker, out, stride, pad, init="gauss(0.01)", decay=1):
    fc = layerprotos.Convolution(ker, out, stride=stride, pad=pad, bias=False)
    fc.param_policies[0] = {"init": init, "decay_mult": decay}
    return fc

def create_model(name="net"):
    main = GModule(name)
    main.input_slots = ('data', 'label')
    inputs = {
        'data' : 'float32(32, 32, 3, _)',
        'label': 'uint32(1, _)'
    }

    x = (main.var('data')
             .to(Convolution(5, 6, stride=1, pad=0),                name='conv1')
             .to(BN(),                                              name='bn1')
             .to(ReLU(), inplace=True,                              name='relu1')
             .to(Pooling('max', 3, stride=2, pad=0),                name='pool1')
             .to(ReLU(), inplace=True,                              name='pool1relu')
             .to(Convolution(5, 16, stride=1, pad=0),               name='conv2')
             .to(BN(),                                              name='bn2')
             .to(ReLU(), inplace=True,                              name='relu2')
             .to(Pooling('max', 3, stride=2, pad=0),                name='pool2')
             .to(ReLU(), inplace=True,                              name='pool2relu')
             .to(FullyConnected(120),                               name='fc1')
             .to(BN(),                                              name='bn3')
             .to(ReLU(), inplace=True,                              name='fc1relu')
             .to(FullyConnected(84),                                name='fc2')
             .to(BN(),                                              name='bn4')
             .to(ReLU(), inplace=True,                              name='fc2relu')
             .to(FullyConnected(10),                                name='fc'))

    main.vars('fc', 'label').to(SoftmaxWithLoss(),                  name='loss')
    main.vars('fc', 'label').to(Accuracy(1),                        name='accuracy_top1')

    return main.compile(inputs=inputs)

if __name__ == '__main__':

    model = create_model()
    print(model.to_yaml_text())

    with open('net.yaml', 'w') as f:
        print >> f, model.to_yaml_text()

    os.system(os.path.join(parrots_home, 'parrots', 'tools', 'visdnn') + ' net.yaml -o net.bn > /dev/null')
