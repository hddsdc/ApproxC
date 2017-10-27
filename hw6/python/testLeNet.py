#######################################################################################
#######       DO NOT MODIFY, DEFINITELY READ THROUGH ALL OF THE CODE            #######
#######################################################################################

import numpy as np
import cnn_lenet
import pickle
import copy
import random

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
  layers[6]['num'] = 500
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


def approximate_and_return(input_list, bits):

    value_index = None
    levels = generate_float_table(bits)

    for i in range(input_list.size):
        minimum_difference = 1
        value = input_list[i]
        for k in range(0, len(levels)):
            temp_difference = value -   levels[k]
            if abs(temp_difference) < minimum_difference:
                minimum_difference = abs(temp_difference)
                value_index = k
        input_list[i] = levels[value_index]

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
    layer1_bias = approximate_and_return(layer1_bias, bits)
    param_grad[1]['b'] = layer1_bias
    # print("Printing Approximate Bias")
    # print(layer1_bias)


def main():
  # define lenet
  layers = get_lenet()

  # load data
  # change the following value to true to load the entire dataset
  fullset = True
  print("Loading MNIST Dataset...")
  xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)
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
  display_interval = 10
  snapshot = 10000
  snapshot2 = 30
  max_iter = 10000
  # max_iter = 500

  bits = 10
  # initialize parameters
  print("Initializing Parameters...")
  params = cnn_lenet.init_convnet(layers)
  param_winc = copy.deepcopy(params)
  print("Initilization Complete!\n")

  print("Generating Approximation Float Table")
  generate_float_table(bits)
  print("Ending Approximation Float Table")

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  random.seed(100000)
  indices = range(m_train)
  random.shuffle(indices)

  print("Training Started. Printing report on training data every " + str(display_interval) + " steps.")
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
