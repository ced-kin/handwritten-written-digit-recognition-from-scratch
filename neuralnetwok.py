import numpy as np


class NeuralNetwork:
    def __init__(self):
        try:
            with np.load("nndata.npz", allow_pickle=True) as network_data:
                self.layer_sizes = network_data["layer_sizes"]
                self.weight_shapes = network_data["weight_shapes"]
                self.weights = network_data["weights"]
                self.biases = network_data["biases"]
                print("nndata.npz file loaded successfully")

        except FileNotFoundError:
            print("nndata.npz file not found, initialising data and creating file")

            self.layer_sizes = (784, 256, 64, 10)
            self.weight_shapes = [(a, b) for a, b in zip(self.layer_sizes[1:], self.layer_sizes[:-1])]
            self.weights = [np.random.standard_normal(s) / (s[1]**.5) for s in self.weight_shapes]
            self.biases = [np.zeros((s, 1)) for s in self.layer_sizes[1:]]

            vals_to_save = {"layer_sizes": self.layer_sizes, "weight_shapes": self.weight_shapes,
                            "weights": self.weights, "biases": self.biases}
            np.savez("nndata.npz", **vals_to_save)

    @staticmethod
    def activation(x):
        return np.maximum(0, x)  # relu function

    @staticmethod
    def activation_prime(x):
        return (x > 0).astype(x.dtype)  # relu prime

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)
        return a

    def feed_forward(self, a):
        activations = [np.empty((s, 1)) for s in self.layer_sizes[1:]]
        z_values = [np.empty((s, 1)) for s in self.layer_sizes[1:]]
        layer_count = 0
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation(z)
            activations[layer_count] = a
            z_values[layer_count] = z
            layer_count += 1
        return activations, z_values

    @staticmethod
    def single_layer_backward_propagation(delta_a, a_prev, w_curr):
        # calculate:
        # dc/da current = 2(a current - true value)
        # dc/db = dcda
        # dc/dw = dcda * a_prev
        # dc/da previous = dcda * weight

        delta_b = delta_a
        delta_w = np.dot(delta_a, np.transpose(a_prev))
        delta_a_prev = np.dot(np.transpose(w_curr), delta_a)
        return delta_b, delta_w, delta_a_prev

    def full_backward_propagation(self, activations, true_value):
        update_weights = [np.empty(s) for s in self.weight_shapes]
        update_biases = [np.empty((s, 1)) for s in self.layer_sizes[1:]]

        delta_a = 2 * np.subtract(activations[-1], true_value)
        for i in range(len(self.weights)):
            temp_delta_b, temp_delta_w, temp_delta_a_prev = self.single_layer_backward_propagation(delta_a,
                                                                                                   activations[-i - 2],
                                                                                                   self.weights[-i - 1])
            update_weights[-i - 1] = temp_delta_w
            update_biases[-i - 1] = temp_delta_b
            delta_a = temp_delta_a_prev
        return update_weights, update_biases

    def update_values(self, update_w, update_b):
        learning_rate = 0.0001
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * update_w[i]
            self.biases[i] -= learning_rate * update_b[i]

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b) for a, b in zip(predictions, labels)])
        print("{0}/{1}  accuracy: {2}%".format(num_correct, len(images), (num_correct / len(images)) * 100))

    def train_network(self, images, labels, epochs):
        for e in range(epochs):
            epoch_cost = 0
            for i in range(len(images)):
                activations, z_values = self.feed_forward(images[i])
                activations.insert(0, images[i])

                prediction = activations[-1]
                cost = np.square(prediction - labels[i])
                epoch_cost += cost.sum()

                uw, ub = self.full_backward_propagation(activations, labels[i])
                self.update_values(uw, ub)

            print(epoch_cost)
            # write the weights and biases to file
            vals_to_save = {"layer_sizes": self.layer_sizes, "weight_shapes": self.weight_shapes,
                            "weights": self.weights, "biases": self.biases}
            np.savez("nndata.npz", **vals_to_save)
