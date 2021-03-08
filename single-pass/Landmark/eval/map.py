import numpy as np

def combine(outputList, sortList):
    CombineList = list();
    for index in range(0, len(outputList)):
        CombineList.append((outputList[index], sortList[index]));
    return CombineList;

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def getlab(strlab):
    index_label = strlab.index(".")
    return strlab[:index_label]

def MAP_test(sess, model, query_feature, query_label, pred_label, unsigned_list_pred_feature, query_size,len_unsigned,len_gan):
    queryrank_out=open('./queryranklabel.txt', 'w+')
    rs = []
    for index_query in range(0, query_size):
        input_query_label = query_label[index_query]
        input_query = []
        input_query.append(query_feature[index_query])
        input_query = np.asarray(input_query)

        queryrank_out.write(input_query_label+"\n")

        input_gan = []
        input_gan_label = []
        for index_gan in range(0, len_unsigned):
            input_gan.append(unsigned_list_pred_feature[index_gan])
            input_gan_label.append(pred_label[index_gan])
        input_gan = np.asarray(input_gan)
        input_gan_label = np.asarray(input_gan_label)

        pred_list_score = sess.run(model.pred_score,feed_dict={model.query_data: input_query, model.pred_data: input_gan})
        """
        exp_rating = np.exp(pred_list_score)
        prob = exp_rating / np.sum(exp_rating)
        sortlist = combine(input_gan_label, prob)
        sortlist.sort(key=lambda x: x[1], reverse=True)
        """
        sortlist = combine(input_gan_label, pred_list_score)
        sortlist.sort(key=lambda x:x[1],reverse=True)

        for i in range (0,11):
            queryrank_out.write(str(i)+" "+str(sortlist[i][0])+"\n")

        r = [0.0] * (len_gan)
        for i in range(0, len_gan):
            input_query_label_str = getlab(input_query_label)
            pred_label_temp = getlab(sortlist[i][0])
            #找到相同类型图片时
            if input_query_label_str == pred_label_temp:
                r[i] = 1.0
        rs.append(r)
    return np.mean([average_precision(r) for r in rs])

def MAP(sess, model, query_feature, query_label, pred_label, generated_data, query_size,len_gan):
    rs = []
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

        pred_list_score = sess.run(model.pred_score, feed_dict={model.query_data: input_query, model.gan_data: input_gan})
        """
        exp_rating = np.exp(pred_list_score)
        prob = exp_rating / np.sum(exp_rating)
        sortlist = combine(input_gan_label, prob)
        sortlist.sort(key=lambda x: x[1], reverse=True)
        """
        sortlist = combine(input_gan_label, pred_list_score )
        sortlist.sort(key=lambda x: x[1], reverse=True)
        r = [0.0] * (len_gan)
        for i in range(0, len_gan):
            input_query_label_str = getlab(input_query_label)
            input_gan_new_label_str = getlab(sortlist[i][0])
            #找到相同类型图片时
            if input_query_label_str == input_gan_new_label_str:
                r[i] = 1.0
        rs.append(r)
    return np.mean([average_precision(r) for r in rs])

def MAP_G(sess, model, query_feature, query_label, pred_label, generated_data, query_size,len_gan):
    rs = []
    for index_query in range(0, query_size):
        input_query_label = query_label[index_query]

        input_gan_label = []
        for index_gan in range(0, len_gan):
            input_gan_label.append(pred_label[int(generated_data[index_query * len_gan + index_gan][1])])
        input_gan_label = np.asarray(input_gan_label)

        r = [0.0] * (len_gan)
        for i in range(0, len_gan):
            input_query_label_str = getlab(input_query_label)
            input_gan_new_label_str = getlab(input_gan_label[i])
            # 找到相同类型图片时
            if input_query_label_str == input_gan_new_label_str:
                r[i] = 1.0
        rs.append(r)
    return np.mean([average_precision(r) for r in rs])