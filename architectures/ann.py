# Implementation of a simple MLP network with one hidden layer. 

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
import random

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 
def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    """ Bias initialization """
    bias = tf.Variable(tf.zeros(shape))
    return bias

def forwardprop(X, w_1, w_2, b_1, b_o):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.relu(tf.add(tf.matmul(X, w_1), b_1))  # The \relu function
    yhat = tf.add(tf.matmul(h, w_2), b_o)
    return yhat

def get_data(nfolds):
    """ Read the joke data set and split them into training and test sets """
    df = pd.read_csv('user_joke_data.csv')
    joke = df["data"]
    target = df["target"]
    
    dataset = np.column_stack((data, target))
    folds = cross_validation_split(dataset, nfolds)
    
    return folds

def main():
    numfolds = 10
    learning_rate = 0.0001
    folds = get_data(numfolds)

    # Layer's sizes
    x_size = 4   # Number of input nodes: number of features
    h_size = 4   # Number of hidden nodes
    y_size = 1   # Number of outcomes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Bias initializations
    b_1 = init_bias(h_size)
    b_o = init_bias(y_size)

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2, b_1, b_o)
    predict = yhat
    
    # Backward propagation
    cost    = tf.reduce_mean(tf.square(tf.subtract(y, yhat)))
    updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    sets_X = list()
    sets_y = list()
    #organizing into training and test sets
    for fold in folds:
        start = [fold[0]]
        for i in range(1,len(folds)):
            start = np.append(start, [fold[i]], axis=0)
        fold = start
        sets_X.append(fold[:,:-1])    #everything before binary encoding
        sets_y.append(fold[:,4:])     #labels 
 
    #build the training and testing sets (data formatting)
    for k in range(1,len(sets_X)):
        train_X = list(sets_X)
        train_y = list(sets_y)
        start_X = train_X[0]
        start_y = train_y[0]
        for j in range(1,len(train_X)):
            start_X = np.append(start_X,train_X[j], axis=0)
            start_y = np.append(start_y,train_y[j], axis=0)
        train_X = start_X
        train_y = start_y
        test_X = sets_X[k]
        test_y = sets_y[k]
    
        for epoch in range(100):
            # Train with each example
            for i in range(len(train_X)): 
                sess.run(updates, feed_dict={X: train_X[i: i+1], y: train_y[i: i + 1]})

				mse = sess.run(cost, feed_dict={X: test_X, y: test_y})
        print('MSE:', mse)
        print('Original: ', test_y[1:2])
        print('Predicted: ', sess.run(predict, feed_dict={X:test_X[1:2]}))
                   

    sess.close()
    

if __name__ == '__main__':
    main()