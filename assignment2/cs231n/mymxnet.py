import find_mxnet
import mxnet as mx

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
