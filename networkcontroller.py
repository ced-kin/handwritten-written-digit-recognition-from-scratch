import neuralnetwok as nn
import numpy as np
import matplotlib.pyplot as plt

with np.load("mnist.npz") as data:
    training_images = data["training_images"]
    training_labels = data["training_labels"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]

net = nn.NeuralNetwork()

# for i in range(100):
#     net.train_network(training_images, training_labels, 1)
#     net.print_accuracy(training_images, training_labels)

# index = 0
# pred = net.predict([training_images[index]])
# print(pred)
# print(np.argmax(pred))
# plt.imshow(training_images[index].reshape(28, 28), cmap="gray")
# plt.show()
# print(np.argmax(training_labels[index]))

net.print_accuracy(training_images, training_labels)
net.print_accuracy(test_images, test_labels)
