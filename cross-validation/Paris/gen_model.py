import tensorflow as tf
import pickle

class GEN:
    def __init__(self, feature_size, weight_decay, learning_rate, len_unsigned, param=None):
        self.feature_size = feature_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.len_unsigned = len_unsigned
        self.g_params = []

        self.reward = tf.placeholder(tf.float32, shape=[None,], name='reward')
        self.sample_index = tf.placeholder(tf.int32, shape=[None, ], name='sample_index')

        self.query_data = tf.placeholder(tf.float32, shape=[1, 2048], name="query_data")
        self.pred_data = tf.placeholder(tf.float32, shape=[self.len_unsigned, 2048], name="pred_data")

        with tf.variable_scope('generator'):
            if param == None:
                self.W_conv = self.weight_variable([1, 1, 2048, self.feature_size])
                self.b_conv = self.bias_variable([self.feature_size])
            else:
                self.W_conv = tf.Variable(param[0])
                self.b_conv = tf.Variable(param[1])
            self.g_params.append(self.W_conv)
            self.g_params.append(self.b_conv)

        #query_vector = tf.reshape(tf.nn.relu(self.conv2d( tf.reshape(self.query_data,[1,1,1,2048]), self.W_conv) + self.b_conv),[1,self.feature_size])
        #pred_vector = tf.reshape(tf.nn.relu(self.conv2d( tf.reshape(self.pred_data,[self.len_unsigned,1,1,2048]), self.W_conv) + self.b_conv),[self.len_unsigned, self.feature_size])

        query_vector = self.query_data
        pred_vector = self.pred_data

        #使用cos距离
        self.pred_score = self.cos_distance(query_vector,pred_vector)

        #使用softmax计算loss
        gan_prob = tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.pred_score, [1, -1])), [-1]), self.sample_index)
        gan_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(gan_prob,1e-8,1.0)) * self.reward) + self.weight_decay * (tf.nn.l2_loss(self.W_conv)+ tf.nn.l2_loss(self.b_conv))

        #随机梯度下降或者Adam算法
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gan_updates = optimizer.minimize(gan_loss, var_list=self.g_params)

    def cos_distance(self,query_vector,pred_vector):
        x3_norm_pred = tf.sqrt(tf.reduce_sum(tf.square(query_vector), 1))
        x4_norm_pred = tf.sqrt(tf.reduce_sum(tf.square(pred_vector), 1))
        x3_x4_pred = tf.reduce_sum(tf.multiply(query_vector,pred_vector), 1)

        cosin_pred = x3_x4_pred / (x3_norm_pred * x4_norm_pred)
        cosin_pred = tf.abs(cosin_pred)
        return cosin_pred

    def save_model(self, sess, filename):
        param = sess.run(self.g_params)
        pickle.dump(param, open(filename, 'wb+'))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')