#######################################################################################
#######       DO NOT MODIFY, DEFINITELY READ THROUGH ALL OF THE CODE            #######
#######################################################################################

import numpy as np
import cnn_lenet
import pickle
import copy
import random
import sys
import getopt

INPUT = 1
WEIGHT = 0

def get_lenet():
  """Define LeNet

  Explanation of parameters:
  type: layer type, supports convolution, pooling, relu
  channel: input channel
  num: output channel
  k: convolution kernel width (== height)
  group: split input channel into several groups, not used in this assignment
  """

  layers = {}
  layers[1] = {}
  layers[1]['type'] = 'DATA'
  layers[1]['height'] = 28
  layers[1]['width'] = 28
  layers[1]['channel'] = 1
  layers[1]['batch_size'] = 64

  layers[2] = {}
  layers[2]['type'] = 'CONV'
  layers[2]['num'] = 20
  layers[2]['k'] = 5
  layers[2]['stride'] = 1
  layers[2]['pad'] = 0
  layers[2]['group'] = 1

  layers[3] = {}
  layers[3]['type'] = 'POOLING'
  layers[3]['k'] = 2
  layers[3]['stride'] = 2
  layers[3]['pad'] = 0

  layers[4] = {}
  layers[4]['type'] = 'CONV'
  layers[4]['num'] = 50
  layers[4]['k'] = 5
  layers[4]['stride'] = 1
  layers[4]['pad'] = 0
  layers[4]['group'] = 1

  layers[5] = {}
  layers[5]['type'] = 'POOLING'
  layers[5]['k'] = 2
  layers[5]['stride'] = 2
  layers[5]['pad'] = 0

  layers[6] = {}
  layers[6]['type'] = 'IP'
  layers[6]['num'] = 50
  layers[6]['init_type'] = 'uniform'

  layers[7] = {}
  layers[7]['type'] = 'RELU'

  layers[8] = {}
  layers[8]['type'] = 'LOSS'
  layers[8]['num'] = 10
  return layers


def generate_float_table(bits):
    total_levels = 1 << (bits-1)
    # print(total_levels)
    levels = []
    # levels.append("Print")
    for i in range(0,2*total_levels):
        value = 1.0*(i/2) / (total_levels)
        if (i % 2):
            value = -value
        levels.append(value)
        # print(levels[i])

    return levels

def generate_float_table_positive(bits):
    total_levels = 1 << (bits)
    # print(total_levels)
    levels = []
    # levels.append("Print")
    for i in range(0,total_levels):
        value = 1.0*(i) / (total_levels)
        levels.append(value)
        # print(levels[i])

    return levels


def approximate_and_return(input_list, bits, input_flag):

    value_index = None
    if input_flag:
        levels = generate_float_table_positive(bits)
    else:
        levels = generate_float_table(bits)

    # print("Input List Size ", input_list.size)
    # print("Input List Shape ", input_list.shape)
    flag = 0
    if input_list.ndim == 2:
        flag = 1
        x_shape = input_list.shape[0]
        y_shape = input_list.shape[1]
        input_list = np.reshape((input_list), x_shape*y_shape)

    for i in range(input_list.size):
        minimum_difference = 3
        value = input_list[i]
        for k in range(0, len(levels)):
            temp_difference = value - levels[k]
            if (abs(temp_difference) < minimum_difference):
                minimum_difference = abs(temp_difference)
                value_index = k
        input_list[i] = levels[value_index]

    if input_list.ndim and flag:
        input_list = np.reshape((input_list), (x_shape, y_shape))
        flag = 0

    # print("Output List Size ", input_list.size)
    # print("Output List Shape ", input_list.shape)
    # print("Value Index:", value_index)
    # print("Minimum Difference:", minimum_difference)
    return input_list


def approx_gradients_step(param_grad, bits, conversion_flag):

    layer1_weights = param_grad[1]['w']
    layer2_weights = param_grad[2]['w']
    layer3_weights = param_grad[3]['w']
    layer4_weights = param_grad[4]['w']
    layer5_weights = param_grad[5]['w']
    layer6_weights = param_grad[6]['w']
    layer7_weights = param_grad[7]['w']

    layer1_bias = param_grad[1]['b']
    layer2_bias = param_grad[2]['b']
    layer3_bias = param_grad[3]['b']
    layer4_bias = param_grad[4]['b']
    layer5_bias = param_grad[5]['b']
    layer6_bias = param_grad[6]['b']
    layer7_bias = param_grad[7]['b']

    # print("Printing Original Bias")
    # print(layer1_bias)
    param_grad[1]['b'] = approximate_and_return(layer1_bias, bits[1], WEIGHT)
    param_grad[2]['b'] = approximate_and_return(layer2_bias, bits[2], WEIGHT)
    param_grad[3]['b'] = approximate_and_return(layer3_bias, bits[3], WEIGHT)
    param_grad[4]['b'] = approximate_and_return(layer4_bias, bits[4], WEIGHT)
    param_grad[5]['b'] = approximate_and_return(layer5_bias, bits[5], WEIGHT)
    param_grad[6]['b'] = approximate_and_return(layer6_bias, bits[6], WEIGHT)
    param_grad[7]['b'] = approximate_and_return(layer7_bias, bits[7], WEIGHT)

    # bits = 9
    param_grad[1]['w'] = approximate_and_return(layer1_weights, bits[1], WEIGHT)
    param_grad[2]['w'] = approximate_and_return(layer2_weights, bits[2], WEIGHT)
    param_grad[3]['w'] = approximate_and_return(layer3_weights, bits[3], WEIGHT)
    param_grad[4]['w'] = approximate_and_return(layer4_weights, bits[4], WEIGHT)
    param_grad[5]['w'] = approximate_and_return(layer5_weights, bits[5], WEIGHT)
    param_grad[6]['w'] = approximate_and_return(layer6_weights, bits[6], WEIGHT)
    param_grad[7]['w'] = approximate_and_return(layer7_weights, bits[7], WEIGHT)
    # print("Printing Approximate Bias")
    # print(layer1_bias)


def main():
  # define lenet
  layers = get_lenet()

  # load data
  # change the following value to true to load the entire dataset
  bits = [ 8, 8, 8, 8, 8, 8, 8, 8]
  max_iter = 2000
  approximation = False
  stepwise_print = False

  # print("Reading command line arguments...")

  try:
    opts, args = getopt.getopt(sys.argv[1:],"h1:2:3:4:5:6:7:i:as")
  except getopt.GetoptError:
    print 'python testLeNet_quant.py -i <training iters> -a -1 <layer1 bits> -2 <layer2 bits> and so on until layer 7'
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
       print '---------------------------------------------------------------------------------------------------------'
       print 'Run Command: python testLeNet.py -a -1 <layer1 bits> -2 <layer2 bits> and so on until layer 7\n'
       print '\t -a | --approx -> for enabling approximation (default: False)'
       print '\t\t Use -a flag and number to specifically tune different layers for different approximations'
       print '\t\t For the current network architecture, use approximations only for layers 1, 3, 5 and 7'
       print '\t\t Other layer approximations do not have any effect as there are no weights for those layers (MAXPOOL/RELU etc)'
       print '\t -s | --stepwise -> for enabling printing of test and training accuracies stepwise (default: False)'
       print '\t -i | --iterations -> for specifying the number of training steps (default: 2000 iterations)'
       print '\n Example Commands: '
       print '\t python testLeNet.py -a -1 6 -3 5 -5 6 -7 6 -s -i 10000'
       print '\t python testLeNet.py -a --l1 6 --l3 5 --l5 6 --l7 6 -s -i 10000'
       print '\t The above commands run 10000 training iterations, with l1 @ 6 bits,'
       print '\t l3 @ 5 bits, l5 @ 6 bits and l7 @ 6 bits with stepwise accuracy printing enabled'
       print '---------------------------------------------------------------------------------------------------------'
       sys.exit()
    elif opt in ('-1', "--l1"):
       bits[1] = int(arg)
    elif opt in ("-2", "--l2"):
       bits[2] = int(arg)
    elif opt in ("-3", "--l3"):
       bits[3] = int(arg)
    elif opt in ("-4", "--l4"):
       bits[4] = int(arg)
    elif opt in ("-5", "--l5"):
       bits[5] = int(arg)
    elif opt in ("-6", "--l6"):
       bits[6] = int(arg)
    elif opt in ("-7", "--l7"):
       bits[7] = int(arg)
    elif opt in ("-i", "--iterations"):
       max_iter = int(arg)
    elif opt in ("-a", "--approx"):
       approximation = True
    elif opt in ("-s", "--stepwise"):
       stepwise_print = True

  print '>> Bit precision of layers [1:7] set to : ', bits
  print '>> Training iterations : ', max_iter
  print '>> Approximation of weights: ', approximation

  # load data
  # change the following value to true to load the entire dataset
  fullset = True
  # print("Loading MNIST Dataset...")

  print("Loading MNIST Dataset...")
  xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)

  # HOOK BEGIN: Change the level of approximation here for Inputs
  # input_bits  = 2
  # xtrain = approximate_and_return(xtrain, input_bits, INPUT)
  # xval   = approximate_and_return(xval, input_bits, INPUT)
  # xtest  = approximate_and_return(xtest, input_bits, INPUT)
  # HOOK END: Change the level of approximation here for Inputs

  #print("xvalidate approximated")
  #print(xval[:,0])
  print("MNIST Dataset Loading Complete!\n")

  xtrain = np.hstack([xtrain, xval])
  ytrain = np.hstack([ytrain, yval])
  m_train = xtrain.shape[1]

  # cnn parameters
  batch_size = 64
  mu = 0.9
  epsilon = 0.01
  gamma = 0.0001
  power = 0.75
  weight_decay = 0.0005
  w_lr = 1
  b_lr = 2

  test_interval = 100
  display_interval = 5
  snapshot = 10000
  snapshot2 = 30
  # max_iter = 10000
  max_iter = 2000

  # initialize parameters
  print("Initializing Parameters...")
  params = cnn_lenet.init_convnet(layers)
  param_winc = copy.deepcopy(params)
  print("Initilization Complete!\n")

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  random.seed(100000)
  indices = range(m_train)
  random.shuffle(indices)

  print(" ---- Training Started ---- \n")
  if stepwise_print:
      print("Printing report on training data every " + str(display_interval) + " steps.")


  print("Printing report on test data every " + str(test_interval) + " steps.\n")
  for step in range(max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step+1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]

    [cp, param_grad] = cnn_lenet.conv_net(params,
                                          layers,
                                          xtrain[:, idx],
                                          ytrain[idx], True)

                                          # we have different epsilons for w and b
    w_rate = cnn_lenet.get_lr(step, epsilon*w_lr, gamma, power)
    b_rate = cnn_lenet.get_lr(step, epsilon*b_lr, gamma, power)

    # print("Printing Parameter gradient")
    # print(param_grad)
    if approximation:
    	approx_gradients_step(param_grad, bits, True)
    # print("Ending Parameter gradient")

    # approx_gradients = param_grad


    params, param_winc = cnn_lenet.sgd_momentum(w_rate,
                           b_rate,
                           mu,
                           weight_decay,
                           params,
                           param_winc,
                           param_grad)

    # display training loss
    if (step+1) % display_interval == 0:
      print 'training_cost = %f training_accuracy = %f' % (cp['cost'], cp['percent']) + ' current_step = ' + str(step + 1)

    # display test accuracy
    if (step+1) % test_interval == 0:
      layers[1]['batch_size'] = xtest.shape[1]
      cptest, _ = cnn_lenet.conv_net(params, layers, xtest, ytest, False)
      layers[1]['batch_size'] = 64
      print 'test_cost = %f test_accuracy = %f' % (cptest['cost'], cptest['percent']) + ' current_step = ' + str(step + 1) + '\n'


    # save params peridocally to recover from any crashes
    if (step+1) % snapshot == 0:
      pickle_path = 'lenet_full.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()

    # save params peridocally to recover from any crashes
    if (((step+1) % snapshot2 == 0) and ((step+1) / snapshot2 == 1)):
      pickle_path = 'lenet_q4.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()


if __name__ == '__main__':
  main()
