# ApproxC

Run Command: python testLeNet.py -a -1 <layer1 bits> -2 <layer2 bits> and so on until layer 7

         -a | --approx -> for enabling approximation (default: False)
                 Use -a flag and number to specifically tune different layers for different approximations
                 For the current network architecture, use approximations only for layers 1, 3, 5 and 7
                 Other layer approximations do not have any effect as there are no weights for those layers (MAXPOOL/RELU etc)

         -s | --stepwise -> for enabling printing of test and training accuracies stepwise (default: False)

         -i | --iterations -> for specifying the number of training steps (default: 2000 iterations)

 Example Commands:
 
 
         python testLeNet.py -a -1 6 -3 5 -5 6 -7 6 -s -i 10000
         
         python testLeNet.py -a --l1 6 --l3 5 --l5 6 --l7 6 -s -i 10000
         
The above commands run 10000 training iterations, with l1 @ 6 bits, l3 @ 5 bits, l5 @ 6 bits and l7 @ 6 bits with stepwise accuracy printing enabled

