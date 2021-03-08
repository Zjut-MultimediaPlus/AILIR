import tensorflow as tf
import pickle
import utils as ut
import numpy as np
from dis_model import DIS
from gen_model import GEN
from eval.map import MAP_test
import time

QUERY_TEST_SIZE = 55
Oxford_PRED_SIZE = 5063
Paris_PRED_SIZE = 6392

LEN_GAN = 400
Oxford_LEN_UNSIGNED = 5063
Paris_LEN_UNSIGNED = 6392

FEATURE_SIZE = 2048
D_WEIGHT_DECAY = 0.0001
G_WEIGHT_DECAY = 0.0001
D_LEARNING_RATE = 0.00003
G_LEARNING_RATE = 0.00003

GAN_MODEL_BEST_FILE = '模型/gan_best.model_2048-2048 0.00003 0.00003 epoch=10 55 TEST 1x1cov all R_mac+RA'

dataset_image_file = 'image_dataset_features_Paris_R_mac+RA.txt'
dataset_label_file = 'label_dataset_features_Paris_R_mac+RA.txt'
query_image_test_file = 'image_query_test_features_Paris_R_mac+RA.txt'
query_label_test_file = 'label_query_test_features_Paris_R_mac+RA.txt'

#所有数据放入list
pred_feature = ut.load_all_pred_feature(dataset_image_file)
query_test_feature = ut.load_all_Paris_query_test_feature(query_image_test_file)

pred_label = ut.load_all_pred_label(dataset_label_file)
query_test_label = ut.load_all_query_label(query_label_test_file)

#转换为numpy矩阵
unsigned_list_pred_feature = np.array(pred_feature)

print("导入数据完成")

#得分与排名两个list的连接
def combine(outputList, sortList):
    CombineList = list()
    for index in range(0, len(outputList)):
        CombineList.append((outputList[index], sortList[index]))
    return CombineList

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print(GAN_MODEL_BEST_FILE)

 # test
    param_best = pickle.load(open(GAN_MODEL_BEST_FILE, 'rb+'), encoding="iso-8859-1")
    assert param_best is not None
    generator_best = GEN(FEATURE_SIZE, G_WEIGHT_DECAY, G_LEARNING_RATE, Paris_LEN_UNSIGNED, param=param_best)
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    map_best = MAP_test(sess, generator_best, query_test_feature, query_test_label, pred_label, pred_feature, QUERY_TEST_SIZE, Paris_LEN_UNSIGNED,LEN_GAN)
    print("GAN_Best MAP ", map_best)
    c=time.ctime()
    print(c)
    sess.close()

if __name__ == '__main__':
    main()