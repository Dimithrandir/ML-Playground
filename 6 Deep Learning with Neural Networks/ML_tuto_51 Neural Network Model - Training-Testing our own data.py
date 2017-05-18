import tensorflow as tf
import numpy as np
import pickle
from create_feature_sets import create_feature_sets_and_labels

# train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt', test_size=0.1)
train_x, train_y, test_x, test_y = pickle.load(open("sentiment_set.pickle","rb"))

# number of nodes for each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# number of classes, also number of output_layer nodes, (pos and neg)
n_classes = 2

# number of feature sets per batch (we need to restrict this so that our memory  can handle it)
# It's no issue for this example, but let's remain 100
batch_size = 100

# input data
x = tf.placeholder('float', [None, len(train_x[0])])
# label of that data 
y = tf.placeholder('float')


def neural_network_model(data):
    # so, each hidden layer is defined by the weight between each of its nodes and previous layer's nodes
    # and each nodes' biases. Biases are used cause most of the time input_value = 0
    # We start with random values at the beggining
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    # now the layers. input_data * weights + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # takes the values and passes through the activation function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    # use cross_entropy_with_logits as our cost function to calculate the difference of our output (prediction)
    # to the labels.
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    # We want to minimaze that cost. We use AdamOptimizer    
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + back propagation
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # for each epoch
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            # we take one batch size per cycle from the training data
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size
                
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)


        # compare the prediction to the actual label
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        # casts var correct to a float
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
        print('Accuracy', accuracy.eval({x:test_x, y:test_y}))
        

train_neural_network(x)



