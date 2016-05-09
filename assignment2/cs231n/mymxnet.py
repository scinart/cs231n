import mxnet as mx
import argparse

def get_args(meta_args):
    args = argparse.Namespace(
        data_dir = '/search/speech/ouyang/opt/mxnet/example/image-classification/cifar10',
        gpus = meta_args['gpus'],
        num_examples=meta_args['num_train'],
        batch_size=meta_args['batch_size'],
        lr = meta_args['learning_rate'],
        lr_factor = meta_args['lr_decay'],
        lr_factor_epoch = 1,
        num_epochs = meta_args['num_epoch'],
        kv_store='local',
        wd=meta_args['reg'],
        std=meta_args['weight_scale'],
        optimizer = meta_args['update_rule'],
        model_prefix=meta_args['model_prefix'],
        load_epoch=meta_args['load_epoch'],
        save_model_prefix=meta_args['save_model_prefix']
    )
    return args

def get_mxnet(mynet, num_classes = 10):
    return get_mxnet_from_mynet(mynet, num_classes)

def get_mxnet_from_mynet(meta_params, num_classes):
    meta_params = meta_params
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
            # bn = mx.symbol.BatchNorm(data=conv)
            bn = conv
            this_layer = act = mx.symbol.Activation(data = bn, act_type='relu', attr=attr)
        elif(layer['type']=='cbv'):
            conv = mx.symbol.Convolution(data=prev_layer,
                                         num_filter=layer['F'],
                                         kernel=(layer['size'],layer['size']),
                                         stride=(layer['stride'],layer['stride']),
                                         pad=(layer['pad'], layer['pad']))
            bn = mx.symbol.BatchNorm(data=conv)
            this_layer = act = mx.symbol.Activation(data = bn, act_type='relu', attr=attr)
        elif(layer['type']=='pl'):
            this_layer = pool = mx.symbol.Pooling(data=prev_layer, kernel=(layer['h'], layer['w']),
                                                  stride=(layer['stride'],layer['stride']), pool_type='max', attr=attr)
        elif (layer['type']=='flat'):
            this_layer = flatten = mx.symbol.Flatten(data=prev_layer, attr=attr)
        elif (layer['type']=='dp'):
            this_layer = dropout = mx.symbol.Dropout(data=prev_layer,p=layer['p'], attr=attr)
        elif (layer['type']=='af'):
            fc = mx.symbol.FullyConnected(data=prev_layer, num_hidden=layer['D'])
            this_layer = fc1relu = mx.symbol.Activation(data = fc, act_type='relu', attr=attr)
        elif (layer['type']=='sbn'):
            this_layer = bn = mx.symbol.BatchNorm(data=prev_layer)
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
