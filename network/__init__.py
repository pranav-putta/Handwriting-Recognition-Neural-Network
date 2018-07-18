# Deployable python file, starts the neural network

from network.Network import Network
import network.mnist_loader
import sys

sys.path.append("../")

print("Starting Propagation...")

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 1.0, test_data)
