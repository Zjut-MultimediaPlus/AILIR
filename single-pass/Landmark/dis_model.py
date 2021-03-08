import tensorflow as tf
import pickle

class DIS():
    def __init__(self, feature_size, weight_decay, learning_rate, loss_name= 'softmax', param=None):
        self.feature_size = feature_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        #self.len_gan = len_gan
        self.d_params = []

        self.query_data = tf.placeholder(tf.float32, shape=[1, 2048], name="query_data")
        self.gan_data = tf.placeholder(tf.float32, shape=[None, 2048], name="gan_data")
        #self.neg_data = tf.placeholder(tf.float32, shape=[self.len_gan, 2048], name="neg_data")

        with tf.variable_scope('discriminator'):
            if param == None:
                self.W_conv = self.weight_variable([1, 1, 2048, self.feature_size])
                self.b_conv = self.bias_variable([self.feature_size])
            else:
                self.W_conv = tf.Variable(param[0])
                self.b_conv = tf.Variable(param[1])
            self.d_params.append(self.W_conv)
            self.d_params.append(self.b_conv)

        query_vector = tf.reshape(tf.nn.relu(self.conv2d(tf.reshape(self.query_data,[1,1,1,2048]), self.W_conv) + self.b_conv),[1,self.feature_size])
        gan_vector = tf.reshape(tf.nn.relu(self.conv2d(tf.reshape(self.gan_data,[-1,1,1,2048]), self.W_conv) + self.b_conv),[-1,self.feature_size])
        #neg_vector = tf.reshape(tf.nn.relu(self.conv2d(tf.reshape(self.neg_data,[self.len_gan,1,1,2048]), self.W_conv) + self.b_conv),[self.len_gan,self.feature_size])

        #query_vector = self.query_data
        #pred_vector = self.pred_data
        #neg_vector = self.neg_data

        #计算距离使用cos距离
        self.pred_score = self.cos_distance(query_vector,gan_vector)
        #self.neg_score = self.cos_distance(query_vector,neg_vector)

        #用距离计算loss，使用softmax
        if loss_name == 'softmax':
            # ranking log loss
            with tf.name_scope('softmax_loss'):
                self.loss = tf.reduce_mean(tf.log(tf.clip_by_value(tf.nn.relu(self.pred_score),1e-8,1.0))) + self.weight_decay * (tf.nn.l2_loss(self.W_conv) + tf.nn.l2_loss(self.b_conv))
                self.reward = tf.reshape(tf.log(tf.clip_by_value(tf.nn.relu(self.pred_score),1e-8,1.0)), [-1])
                #self.loss = -tf.reduce_mean(tf.log(tf.nn.relu(self.pred_score))) + self.weight_decay * (tf.nn.l2_loss(self.W_conv) + tf.nn.l2_loss(self.b_conv))
                #self.reward = tf.reshape(tf.log(tf.nn.relu(self.pred_score)), [-1])
        else:
            assert 'You should use svm and softmax.'

        #随机梯度下降或者Adam算法
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #optimizer =tf.train.RMSPropOptimizer(self.learning_rate)
        self.d_updates = optimizer.minimize(self.loss, var_list=self.d_params)

    def cos_distance(self,query_vector,gan_vector):
        x3_norm_pred = tf.sqrt(tf.reduce_sum(tf.square(query_vector), 1))
        x4_norm_pred = tf.sqrt(tf.reduce_sum(tf.square(gan_vector), 1))
        x3_x4_pred = tf.reduce_sum(tf.multiply(query_vector,gan_vector), 1)

        cosin_pred = x3_x4_pred / (x3_norm_pred * x4_norm_pred)
        cosin_pred = tf.abs(cosin_pred)
        return cosin_pred

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        pickle.dump(param, open(filename, 'wb+'))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')