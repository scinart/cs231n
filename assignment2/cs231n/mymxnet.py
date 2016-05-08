import find_mxnet
import mxnet as mx

def get_test(num_classes = 10, force_mirroring=False):
    if force_mirroring:
        attr = {'force_mirroring': 'true'}
    else:
        attr = {}

    data = mx.symbol.Variable(name="data")
    data = mx.symbol.Convolution(data=data, num_filter=32, kernel=(7,7), stride=(1,1), pad=(3,3))
    data = mx.symbol.BatchNorm(data=data)
    data = mx.symbol.Activation(data=data, act_type='relu', attr=attr)
    data = mx.symbol.Pooling(data=data, kernel=(2, 2), stride=(2, 2), pool_type='max', attr=attr)
    data = mx.symbol.Flatten(data=data, attr=attr)
    data = mx.symbol.FullyConnected(data=data, num_hidden=100, name="fc1")
    data = mx.symbol.Activation(data=data, act_type='relu', attr=attr)
    data = mx.symbol.FullyConnected(data=data, num_hidden=num_classes, name="fc2")
    data = mx.symbol.SoftmaxOutput(data=data, name="softmaxeeee")
    return data

def get_mxnet(num_classes = 10):

    net1 = [ {'type': 'cv', 'F': 32, 'size':7, 'stride': 1, 'pad': 3},
             {'type': 'pl', 'h':2, 'w':2, 'stride':2},
             {'type': 'flat'},
             {'type': 'af', 'D': 100},
             {'type': 'sm', 'num_class':10} ]

    meta_params = net1
    num_layers = len(meta_params)
    attr = {}
    data = mx.symbol.Variable(name="data")
    net = {}
    net['layer'+str(0)]=data
    for i in range(1,num_layers+1):
        layer=meta_params[i-1]
        prev_layer = net['layer'+str(i-1)]
        if(layer['type']=='cv'):
            conv = mx.symbol.Convolution(data=prev_layer,
                                         num_filter=layer['F'],
                                         kernel=(layer['size'],layer['size']),
                                         stride=(layer['stride'],layer['stride']),
                                         pad=(layer['pad'], layer['pad']))
            bn = mx.symbol.BatchNorm(data=conv)
            this_layer = act = mx.symbol.Activation(data = bn, act_type='relu', attr=attr)
        elif(layer['type']=='pl'):
            this_layer = pool = mx.symbol.Pooling(data=prev_layer, kernel=(layer['h'], layer['w']), stride=(layer['stride'],layer['stride']), pool_type='max', attr=attr)
        elif (layer['type']=='flat'):
            this_layer = flatten = mx.symbol.Flatten(data=prev_layer, attr=attr)
        elif (layer['type']=='af'):
            fc = mx.symbol.FullyConnected(data=prev_layer, num_hidden=layer['D'])
            this_layer = fc1relu = mx.symbol.Activation(data = fc, act_type='relu', attr=attr)
        elif (layer['type']=='sm'):
            fc = mx.symbol.FullyConnected(data=prev_layer, num_hidden=num_classes)
            this_layer = softmax = mx.symbol.SoftmaxOutput(data=fc, name="softmax")

        net['layer'+str(i)]=this_layer

    return softmax
    # return net['layer'+str(num_layers)]

def get_cifar_default(num_classes = 10, force_mirroring=False):
    if force_mirroring:
        attr = {'force_mirroring': 'true'}
    else:
        attr = {}

    data = mx.symbol.Variable(name="data")
    conv = mx.symbol.Convolution(data=data, num_filter=32, kernel=(7,7), stride=(1,1), pad=(3,3))
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type='relu', attr=attr)
    pool = mx.symbol.Pooling(data=act, kernel=(2, 2), stride=(2, 2), pool_type='max', attr=attr)
    flatten = mx.symbol.Flatten(data=pool, name="flatten1", attr=attr)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100, name="fc1")
    fc1relu = mx.symbol.Activation(data = fc1, act_type='relu', attr=attr)
    fc2 = mx.symbol.FullyConnected(data=fc1relu, num_hidden=num_classes, name="fc2")
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name="softmax")
    return softmax
