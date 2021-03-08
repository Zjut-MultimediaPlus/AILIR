import tensorflow as tf
import pickle
import utils as ut
import numpy as np
import random
from dis_model import DIS
from gen_model import GEN
from eval.map import MAP
from eval.map import MAP_G
from eval.map import MAP_test

#每个epoch拥有55个query
QUERY_TRAIN_SIZE = 55  #batch_size 55
QUERY_TEST_SIZE = 55
TRAIN_SIZE = 6000 #258332

LEN_GAN = 400
Landmark_LEN_UNSIGNED = 6000 #258332 #5063
Oxford_LEN_UNSIGNED = 5063
Paris_LEN_UNSIGNED = 6392

FEATURE_SIZE = 2048
D_WEIGHT_DECAY = 0.0001
G_WEIGHT_DECAY = 0.0001
D_LEARNING_RATE = 0.000005
G_LEARNING_RATE = 0.000001

#GAN_MODEL_FILE = 'Landmark_gan.model_2048-2048 0.0005 0.00001'
GAN_MODEL_BEST_FILE = 'Landmark_gan_best.model_2048-2048 0.000005 0.000001'
#GAN_MODEL_BEST_FILE_OLD = ''

Landmark_dataset_image_file = '6k_image_dataset_features_landmark.txt'
Landmark_dataset_index_file = '6k_index_dataset_features_landmark.txt'

Oxford_dataset_image_file = 'image_dataset_features_Oxford_R_mac+RA.txt'
Oxford_dataset_label_file = 'label_dataset_features_Oxford_R_mac+RA.txt'
Oxford_query_image_test_file = 'image_query_test_features_Oxford_R_mac+RA.txt'
Oxford_query_label_test_file = 'label_query_test_features_Oxford_R_mac+RA.txt'

#所有数据放入list
Landmark_pred_feature = ut.load_all_pred_feature(Landmark_dataset_image_file)
Landmark_pred_rank = ut.load_all_pred_rank(Landmark_dataset_index_file)

Oxford_pred_feature = ut.load_all_pred_feature(Oxford_dataset_image_file)
Oxford_pred_label = ut.load_all_pred_label(Oxford_dataset_label_file)
Oxford_query_test_feature = ut.load_all_Oxford_query_test_feature(Oxford_query_image_test_file)
Oxford_query_test_label = ut.load_all_query_label(Oxford_query_label_test_file)

Landmark_query_train_feature = Landmark_pred_feature
#转换为numpy矩阵
Landmark_unsigned_list_pred_index = np.array(Landmark_pred_rank)


#得分与排名两个list的连接
def combine(outputList, sortList):
    CombineList = list()
    for index in range(0, len(outputList)):
        CombineList.append((outputList[index], sortList[index]))
    return CombineList

def main():
    print("load initial model ...")
    #param_old = pickle.load(open(GAN_MODEL_BEST_FILE_OLD, 'rb+'), encoding="iso-8859-1")
    #generator_best = GEN(FEATURE_SIZE, G_WEIGHT_DECAY, G_LEARNING_RATE, Oxford_LEN_UNSIGNED, param=None)

    generator = GEN(FEATURE_SIZE, G_WEIGHT_DECAY, G_LEARNING_RATE, param=None)
    print('Gen Done!!!')
    discriminator = DIS(FEATURE_SIZE, D_WEIGHT_DECAY, D_LEARNING_RATE, param=None)
    print("DIS Done!!!")

    Test_map_best = 0

    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(GAN_MODEL_BEST_FILE)

    for epoch in range(40):
        print("epoch" + str(epoch))
        # 从PRED_SIZE中随机抽取QUERY_TRAIN_SIZE个样本作为query
        random_query_D_feature = []
        #random_query_D_label = []
        generated_data = []
        for index_query in range(0, QUERY_TRAIN_SIZE):
            if index_query % 10 == 0 or index_query == QUERY_TRAIN_SIZE:
                print("random_query_from_G_for_D " + str(index_query))
            # 随机生成query序号
            query = random.randint(0, TRAIN_SIZE - 1)
            random_query_D_feature.append(Landmark_query_train_feature[query])
            #random_query_D_label.append(query_train_label[query])

            current_query_feature = []
            current_query_feature.append(Landmark_query_train_feature[query])
            current_query_feature = np.asarray(current_query_feature)

            # 针对每一个query，计算dataset的得分以及根据softmax对dataset排序
            pred_list_score = sess.run(generator.pred_score, feed_dict={generator.query_data: current_query_feature, generator.pred_data: Landmark_pred_feature})

            exp_rating = np.exp(pred_list_score)
            prob = exp_rating / np.sum(exp_rating)
            sortlist = combine(Landmark_unsigned_list_pred_index, prob)
            sortlist.sort(key=lambda x: x[1], reverse=True)
            # 取排名的前LEN_GAN个加入generated_data  query序号 + dataset图片序号 + dataset特征
            for i in range(0, LEN_GAN):
                generated_data.append((index_query, sortlist[i][0], Landmark_pred_feature[int(sortlist[i][0])]))

        # Train D
        print('Training D ...')
        for d_epoch in range(30):
            dis_train_loss_sum = 0
            print('d_epoch'+str(d_epoch))
            for index_query in range(0, QUERY_TRAIN_SIZE):
                input_query=[]
                input_query.append(random_query_D_feature[index_query])
                input_gan = []
                for index_gan in range(0, LEN_GAN):
                    input_gan.append(generated_data[index_query * LEN_GAN + index_gan][2])
                _ = sess.run(discriminator.d_updates,feed_dict={discriminator.query_data: input_query, discriminator.gan_data: input_gan})
                dis_train_loss = sess.run(discriminator.loss, feed_dict={discriminator.query_data: input_query, discriminator.gan_data: input_gan})
                dis_train_loss_sum += dis_train_loss
            dis_train_loss_mean = dis_train_loss_sum/QUERY_TRAIN_SIZE
            print("epoch:", epoch, "/ 40", "d_epoch:", d_epoch, "/ 30", "dis_train_loss_mean:",dis_train_loss_mean)

        # Train G
        print('Training G ...')
        number_index = np.random.permutation(TRAIN_SIZE)
        number = 0
        for g_epoch in range(30):
            print('g_epoch'+str(g_epoch))
            #从PRED_SIZE中随机抽取QUERY_TRAIN_SIZE个样本作为query
            random_query_G_feature=[]
            #random_query_G_label=[]
            generated_data=[]
            gen_train_loss_sum = 0
            for index_query in range(0, QUERY_TRAIN_SIZE):
                if index_query % 10 == 0 or index_query == QUERY_TRAIN_SIZE:
                    print("random_query_from_G_for_G " + str(index_query))

                # 随机生成query序号
                if number == TRAIN_SIZE - 1:
                    number = 0
                query = number_index[number]
                number = number + 1
                random_query_G_feature.append(Landmark_query_train_feature[query])
                #random_query_G_label.append(query_train_label[query])

                current_query_feature_un = []
                current_query_feature_un.append(Landmark_query_train_feature[query])
                current_query_feature_un = np.asarray(current_query_feature_un)

                #针对每一个query，计算dataset的得分以及根据softmax对dataset排序
                pred_list_score = sess.run(generator.pred_score,feed_dict={generator.query_data: current_query_feature_un, generator.pred_data: Landmark_pred_feature})

                exp_rating = np.exp(pred_list_score)
                prob = exp_rating / np.sum(exp_rating)
                sortlist = combine(Landmark_unsigned_list_pred_index, prob)
                sortlist.sort(key=lambda x: x[1], reverse=True)
                # 取排名的前LEN_GAN个加入generated_data  query序号 + dataset图片序号 + dataset特征
                for i in range(0, LEN_GAN):
                    generated_data.append((index_query, sortlist[i][0], Landmark_pred_feature[int(sortlist[i][0])]))
                #获取根据query检索出来的图库中的图片特征
                gan_list_feature = []
                for index_gan in range(0, LEN_GAN):
                    gan_list_feature.append(generated_data[index_query * LEN_GAN + index_gan][2])
                gan_list_feature = np.asarray(gan_list_feature)

                #根据生成的GAN序列和query的特征进行生成reward
                gan_reward = sess.run(discriminator.reward, feed_dict={discriminator.query_data: current_query_feature_un,discriminator.gan_data: gan_list_feature})
                gan_index = np.random.choice(np.arange(len(Landmark_unsigned_list_pred_index)),size=LEN_GAN,p=prob)

                _ = sess.run(generator.gan_updates, feed_dict={generator.query_data: current_query_feature_un,generator.pred_data: Landmark_pred_feature, generator.sample_index: gan_index, generator.reward: gan_reward})
                gen_train_loss = sess.run(generator.gan_loss, feed_dict={generator.query_data: current_query_feature_un,generator.pred_data: Landmark_pred_feature, generator.sample_index: gan_index, generator.reward: gan_reward})
                gen_train_loss_sum += gen_train_loss

            gen_train_loss_mean = gen_train_loss_sum/QUERY_TRAIN_SIZE
            print("epoch:", epoch, "/ 40", "g_epoch:", g_epoch, "/ 30", "gen_train_loss:", gen_train_loss_mean)

            Test_map = MAP_test(sess, generator, Oxford_query_test_feature, Oxford_query_test_label, Oxford_pred_label, Oxford_pred_feature, QUERY_TEST_SIZE, Oxford_LEN_UNSIGNED, LEN_GAN)
            print("Test_map:", Test_map)
            if Test_map > Test_map_best:
                Test_map_best = Test_map
                generator.save_model(sess, GAN_MODEL_BEST_FILE)
                print("Best_Test_map:", Test_map_best)

    # test
    # param_best = pickle.load(open(GAN_MODEL_BEST_FILE, 'rb+'), encoding="iso-8859-1")
    # assert param_best is not None
    # generator_best = GEN(FEATURE_SIZE, G_WEIGHT_DECAY, G_LEARNING_RATE, LEN_UNSIGNED, param=param_best)
    # sess = tf.Session(config=config)
    # sess.run(tf.initialize_all_variables())
    # map_best = MAP_test(sess, generator_best, query_test_feature, query_test_label, pred_label, pred_feature, QUERY_TEST_SIZE, LEN_UNSIGNED,LEN_GAN)
    # print("GAN_Best MAP ", map_best)
    print("Train End")
    print(Test_map_best)
    sess.close()

if __name__ == '__main__':
    main()
