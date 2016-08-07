from numpy import *


# input format : dot(transpose(w), x)
# w = (w0, w1, w2, ...)
# x = (1, x1, x2, ...)
def sigmoid(x):
    return 1/(1+exp(-x))


def sigmoid_differential(x):
    return x*(1-x)


# gradient descent
# activation function: f(x)
# cost function for pattern mode: J(w) = (y_i - f(dot(transpose(w), X_i)))^2
# partial differential respect to w: J'(w) = -2*error*output*(1-output)*x_i
# w(t): weights on t-th iteration
# w(t+1) = w(t) - learning rate*J'(w)
def calculate_gradient(y, x_vector, weight_vector, learning_rate):
    sigmoid_output = sigmoid(dot(weight_vector.T, x_vector))
    increment = -2*(y-sigmoid_output)*sigmoid_differential(sigmoid_output)
    weight_vector -= learning_rate*increment
    return weight_vector


def train_neural_network(data_matrix, labels, iterator_number):
    training_data_size = shape(data_matrix)[0]
    weight_vector_length = shape(data_matrix)[1]
    weight_vector = 2 * random.random(weight_vector_length) - 1
    for i in range(iterator_number):
        for j in range(training_data_size):
            data = array(data_matrix[j]).flatten()
            weight_vector = calculate_gradient(data, labels[j], weight_vector, 0.5)
            print "weight_vector: ", weight_vector
    return weight_vector


def get_training_set():
    data_matrix = mat([[1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [1, 1, 0, 1],
                      [1, 0, 1, 1]])
    labels = array([0, 1, 1, 0]).T
    return data_matrix, labels


#input data format: array
def neural_network_classifier(input_data):
    data_matrix, labels = get_training_set()
    weight_vector = train_neural_network(data_matrix, labels, 40)
    result = sigmoid(dot(weight_vector.T, input_data))
    print "result : %.5f" % result
