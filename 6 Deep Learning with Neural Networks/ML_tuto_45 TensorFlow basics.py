#! python3.5

import tensorflow as tf

if __name__ == '__main__':

    x1 = tf.constant(5)
    x2 = tf.constant(6)

    result = tf.multiply(x1,x2)

    with tf.Session() as sess:
        print(sess.run(result))
