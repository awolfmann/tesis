caffe_root = '/home/ariel/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *
import tempfile
import matplotlib
import os
from os.path import join, exists
# import cv2
import csv
import glob



# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

def process_image(image):
    image = image.copy()              # don't modify destructively
    image = caffe.io.resize_image( image, (227,227), interp_order=3 ) # RESCALE
    image -= [123, 117, 104]          # (approximately) mean subtraction
    image = image[::-1]               # RGB -> BGR
    image = image.transpose(2, 0, 1)  # HWC -> CHW   
    # image -= [104, 117, 123]          # (approximately) mean subtraction

    # clamp values in [0, 255]
    # image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    # image = np.round(image)
    image = np.require(image, dtype=np.float32)

    return image

# weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# weights = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style.caffemodel'
weights = caffe_root + 'models/finetune_wikipaintings_style/finetune_wikipaintings_style_iter_10000.caffemodel'
assert os.path.exists(weights)

# Load ImageNet labels to imagenet_labels
# imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
# imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
# assert len(imagenet_labels) == 1000
# print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

# Load style labels to style_labels
style_label_file = caffe_root + 'data/wikipaintings_style/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
print '\nLoaded style labels:\n', ', '.join(style_labels)

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name

# dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
# imagenet_net_filename = caffenet(data=dummy_data, train=False)
# imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

def style_net(train=True, learn_all=False, subset=None):
    # if subset is None:
    #     subset = 'train' if train else 'test'
    source = caffe_root + 'data/wikipaintings_style/subset_wikipaintings.txt'
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=1, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=10,
                    classifier_name='fc8_wikipaintings',
                    learn_all=learn_all)

untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
                                weights, caffe.TEST)
untrained_style_net.forward()
style_data_batch = untrained_style_net.blobs['data'].data.copy()
style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)

del untrained_style_net


def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 5000
    s.snapshot_prefix = caffe_root + 'models/finetune_wikipaintings_style/finetune_wikipaintings_style'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    #s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

def run_solvers(niter, solvers, disp_interval=50):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

niter = 30000  # number of iterations to train

# Reset style_solver as before.
# style_solver_filename = solver(style_net(train=True))
# style_solver = caffe.get_solver(style_solver_filename)
# style_solver.net.copy_from(weights)

# For reference, we also create a solver that isn't initialized from
# the pretrained ImageNet weights.
# scratch_style_solver_filename = solver(style_net(train=True))
# scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)

# print 'Running solvers for %d iterations...' % niter
# # solvers = [('pretrained', style_solver),
# #            ('scratch', scratch_style_solver)]
# solvers = [('pretrained', style_solver)]
# loss, acc, weights = run_solvers(niter, solvers)
# print 'Done.'

# train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
# train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
# style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']
# train_loss = loss['pretrained']
# train_acc = acc['pretrained']
# style_weights = weights['pretrained']


# # Delete solvers to save memory.
# del style_solver, solvers

# end_to_end_net = style_net(train=True, learn_all=True)

# Set base_lr to 1e-3, the same as last time when learning only the classifier.
# You may want to play around with different values of this or other
# optimization parameters when fine-tuning.  For example, if learning diverges
# (e.g., the loss gets very large or goes to infinity/NaN), you should try
# decreasing base_lr (e.g., to 1e-4, then 1e-5, etc., until you find a value
# for which learning does not diverge).
base_lr = 1e-4

# style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
# style_solver = caffe.get_solver(style_solver_filename)
# style_solver.net.copy_from(weights)
# solverstate = caffe_root + 'models/finetune_wikipaintings_style/finetune_wikipaintings_style_iter_10000.solverstate'
# assert os.path.exists(solverstate)
# style_solver.restore(solverstate)

# scratch_style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
# scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)
# scratch_style_solver.net.copy_from(scratch_style_weights)

# print 'Running solvers for %d iterations...' % niter
# solvers = [('pretrained, end-to-end', style_solver)]
# _, _, finetuned_weights = run_solvers(niter, solvers)
# print 'Done.'

# style_weights_ft = finetuned_weights['pretrained, end-to-end']
# scratch_style_weights_ft = finetuned_weights['scratch, end-to-end']

# Delete solvers to save memory.
# del style_solver, solvers

weights1 = caffe_root + 'models/finetune_wikipaintings_style/finetune_wikipaintings_style_iter_40000.caffemodel'
assert os.path.exists(weights1)
style_net_loaded = caffe.Net(style_net(train=False, subset='test'),
                                weights1, caffe.TEST)

# batch_index = 8
# image = style_data_batch[batch_index]
# plt.imshow(deprocess_net_image(image))
# print 'actual label =', style_labels[style_label_batch[batch_index]]
# plt.imshow(deprocess_net_image(image))
del style_data_batch

# image_filename = '/home/ariel/Documents/tesis/neural-style/examples/inputs/picasso_selfport1907.jpg' 


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': style_net_loaded.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR  
style_net_loaded.blobs['data'].reshape( 1,        # batch size
                                        3,         # 3-channel (BGR) images
                                        227, 227)  # image size is 227x227

# # image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# plt.imshow(image)
## copy the image data into the memory allocated for the net
# net.blobs['data'].data[...] = transformed_image

def disp_preds(net, image, labels, file, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    # net.blobs['data'].data[0, ...] = image
    net.blobs['data'].data[...] = image

    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    if file:
        file.write('top %d predicted %s labels =' % (k, name))
        file.write('\n')
        file.write('\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                        for i, p in enumerate(top_k)))
        file.write('\n')
    else:
        print('top %d predicted %s labels =' % (k, name))
        print('\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                        for i, p in enumerate(top_k)))

    
def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image, file=None):
    disp_preds(net, image, style_labels, file, name='style')

def predictions(net, image):
    input_blob = net.blobs['data']
    net.blobs['data'].data[...] = image
    probs = net.forward(start='conv1')['probs'][0]

    return probs

# output_path ='/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/'
# # scores_file = open(join(output_path,'scores.txt'), 'w')

# content_image_file = '/home/ariel/Documents/tesis/neural-style/examples/inputs/brad_pitt.jpg'
# content_image = caffe.io.load_image(content_image_file)
# transformed_content_image = transformer.preprocess('data', content_image)
# # scores_file.write('CONTENT IMAGE \n')
# # disp_style_preds(style_net_loaded, transformed_content_image, scores_file)

# style_image_file = '/home/ariel/Documents/tesis/neural-style/examples/inputs/woman-with-hat-matisse.jpg' 
# style_image = caffe.io.load_image(style_image_file)
# transformed_style_image = transformer.preprocess('data', style_image)
# scores_file.write('STYLE IMAGE \n')
# disp_style_preds(style_net_loaded, transformed_style_image, scores_file)

def process_scores(output_path, output_name):
    with open(join(output_path,'scores.csv'), 'wb') as csvfile_writer:
        csvwriter = csv.writer(csvfile_writer, delimiter=',')
        csvwriter.writerow(['iteration'] + style_labels)
        # if start == 'image':
        #     probs = predictions(style_net_loaded, transformed_content_image)
        #     csvwriter.writerow([0] + probs.tolist())
        # elif start == 'random':
        #     probs = predictions(style_net_loaded, transformed_style_image)
        #     csvwriter.writerow([0] + probs.tolist()) 
        # else:
        #     print 'ERROR'           
        for it in range(0, 1001, 10):
            outname = output_name +'_{:d}.png'.format(it)
            image_filename = join(output_path, outname)
            image = caffe.io.load_image(image_filename)
            transformed_image = transformer.preprocess('data', image)
            probs = predictions(style_net_loaded, transformed_image)
            csvwriter.writerow([it] + probs.tolist())

# process_scores(start='IMAGE')

def tag_style_images():
    print('TAGS')
    for image_filename in glob.iglob('/home/ariel/Documents/tesis/tests/inputs/style_images/*'):
        print('FILENAME ', image_filename)
        image = caffe.io.load_image(image_filename)
        transformed_image = transformer.preprocess('data', image)
        disp_style_preds(style_net_loaded, transformed_image)

# tag_style_images()

# transfered_image_file = '/home/ariel/Documents/tesis/neural-style/examples/outputs/golden_gate_kahlo.png' 
# transfered_image = caffe.io.load_image(transfered_image_file)
# transformed_transfered_image = transformer.preprocess('data', transfered_image)
# print('TRANSFERED IMAGE')
# disp_style_preds(style_net_loaded, transformed_transfered_image)


# output_path ='/home/ariel/Documents/tesis/neural-style/examples/outputs/sculpture/'

# for it in range(10, 990, 10):
#     image_filename = join(output_path, 'pitt_matisse_{:d}.png'.format(it))
#     scores_file.write('ITERATIONS {} \n'.format(it))
#     image = caffe.io.load_image(image_filename)
#     transformed_image = transformer.preprocess('data', image)
#     disp_style_preds(style_net_loaded, transformed_image, scores_file)


