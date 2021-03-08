import numpy as np

def combine(outputList, sortList):
    CombineList = list()
    for index in range(0, len(outputList)):
        CombineList.append((outputList[index], sortList[index]))
    return CombineList

def compute_ap(rank_list,pos_set_label,pos_set_label_num):
    #rank_list:查询图像返回的结果
    #pos_list：数据库张与查询图像相似的结果
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0.0
    for i in range(len(rank_list)):
        if rank_list[i] == pos_set_label:
            intersect_size +=1
            recall = intersect_size / pos_set_label_num
            precision = intersect_size / (i+1)
            ap += (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision
    return ap

def getlab(strlab):
    index_label = strlab.index(".")
    return strlab[:index_label]

def MAP_test(sess, model, query_feature, query_label, pred_label, unsigned_list_pred_feature, query_size,len_unsigned,len_gan):
    ap_sum = 0
    for index_query in range(0, query_size):
        input_query_label = query_label[index_query]
        input_query = []
        input_query.append(query_feature[index_query])
        input_query = np.asarray(input_query)

        input_gan = []
        input_gan_label = []
        for index_gan in range(0, len_unsigned):
            input_gan.append(unsigned_list_pred_feature[index_gan])
            input_gan_label.append(pred_label[index_gan])
        input_gan = np.asarray(input_gan)
        input_gan_label = np.asarray(input_gan_label)

        pred_list_score = sess.run(model.pred_score,feed_dict={model.query_data: input_query, model.gan_data: input_gan})
        """
        exp_rating = np.exp(pred_list_score)
        prob = exp_rating / np.sum(exp_rating)
        sortlist = combine(input_gan_label, prob)
        sortlist.sort(key=lambda x: x[1], reverse=True)
        """
        sortlist = combine(input_gan_label, pred_list_score)
        sortlist.sort(key=lambda x:x[1],reverse=True)

        input_query_label_str = getlab(input_query_label)
        pos_set_label_num = 0
        for i in range(0,len(pred_label)):
            if getlab(pred_label[i]) == input_query_label_str:
                pos_set_label_num += 1
        rank_list = {}
        for i in range(0,len_gan):
            rank_list[i] = getlab(sortlist[i][0])

        ap = compute_ap(rank_list,input_query_label_str,pos_set_label_num)
        ap_sum += ap
    map = ap_sum / query_size
    return map

def MAP_D(sess, model, query_feature, query_label, pred_label, generated_data, query_size,len_gan):
    ap_sum = 0
    for index_query in range(0, query_size):
        input_query_label = query_label[index_query]
        input_query = []
        input_query.append(query_feature[index_query])
        input_query = np.asarray(input_query)

        input_gan = []
        input_gan_label = []
        for index_gan in range(0,len_gan):
            input_gan.append(generated_data[index_query*len_gan+index_gan][2])
            input_gan_label.append(pred_label[int(generated_data[index_query*len_gan+index_gan][1])])
        input_gan = np.asarray(input_gan)
        input_gan_label = np.asarray(input_gan_label)

        pred_list_score = sess.run(model.pred_score, feed_dict={model.query_data: input_query, model.pred_data: input_gan})
        """
        exp_rating = np.exp(pred_list_score)
        prob = exp_rating / np.sum(exp_rating)
        sortlist = combine(input_gan_label, prob)
        sortlist.sort(key=lambda x: x[1], reverse=True)
        """
        sortlist = combine(input_gan_label, pred_list_score )
        sortlist.sort(key=lambda x: x[1], reverse=True)

        input_query_label_str = getlab(input_query_label)
        pos_set_label_num = 0
        for i in range(0, len(pred_label)):
            if getlab(pred_label[i]) == input_query_label_str:
                pos_set_label_num += 1
        rank_list = {}
        for i in range(0, len_gan):
            rank_list[i] = getlab(sortlist[i][0])
        ap = compute_ap(rank_list, input_query_label_str, pos_set_label_num)
        ap_sum += ap
    map = ap_sum / query_size
    return map

def MAP_G(sess, model, query_feature, query_label, pred_label, generated_data, query_size,len_gan):
        ap_sum = 0
        for index_query in range(0, query_size):
            input_query_label = query_label[index_query]
            input_query = []
            input_query.append(query_feature[index_query])
            input_query = np.asarray(input_query)

            input_gan = []
            input_gan_label = []
            for index_gan in range(0, len_gan):
                input_gan.append(generated_data[index_query * len_gan + index_gan][2])
                input_gan_label.append(pred_label[int(generated_data[index_query * len_gan + index_gan][1])])
            input_gan = np.asarray(input_gan)
            input_gan_label = np.asarray(input_gan_label)

            pred_list_score = sess.run(model.pred_score,feed_dict={model.query_data: input_query, model.gan_data: input_gan})
            """
            exp_rating = np.exp(pred_list_score)
            prob = exp_rating / np.sum(exp_rating)
            sortlist = combine(input_gan_label, prob)
            sortlist.sort(key=lambda x: x[1], reverse=True)
            """
            sortlist = combine(input_gan_label, pred_list_score)
            sortlist.sort(key=lambda x: x[1], reverse=True)

            input_query_label_str = getlab(input_query_label)
            pos_set_label_num = 0

            for i in range(0, len(pred_label)):
                if getlab(pred_label[i]) == input_query_label_str:
                    pos_set_label_num += 1
            rank_list = {}
            for i in range(0, len_gan):
                rank_list[i] = getlab(sortlist[i][0])

            ap = compute_ap(rank_list, input_query_label_str, pos_set_label_num)
            ap_sum += ap
        map = ap_sum / query_size
        return map
