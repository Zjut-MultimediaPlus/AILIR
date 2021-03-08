import numpy as np

def combine(outputList,sortList):  
    CombineList = list();
    for index in range(0, len(outputList)):  
        CombineList.append((outputList[index],sortList[index]));  
    return CombineList; 

def precision_at_k(sess, model, query_feature, query_label, pred_label, unsigned_list_pred_feature, query_size, len_unsigned, k):
    p = 0.0
    cnt = 0
    ap=0.0
    for index_query in range(0, query_size):
        input_query = []

        input_query_label = query_label[index_query]
        input_gan = []
        input_gan_label = []
        #pred_label_500 = []
        for index_gan in range(0,len_unsigned):
            input_query.append(query_feature[index_query])
            input_gan.append(unsigned_list_pred_feature[index_gan])
            input_gan_label.append(pred_label[index_gan])
            #print pred_label[generated_data[index_query*len_gan+index_gan][1]]
            #input_gan_label.append(pred_label[generated_data[index_gan][1]])
        input_query = np.asarray(input_query)
        input_gan = np.asarray(input_gan)
        input_gan_label = np.asarray(input_gan_label)
        pred_list_score = sess.run(model.pred_score, feed_dict={model.query_data: input_query, model.gan_data: input_gan})
     
        sortlist = combine(input_gan_label, pred_list_score)
        #print sortlist      
        sortlist.sort(key=lambda x:x[1],reverse=True);  
        #print sortlist;
   
        num = 0.0
        num_p = 0.0

        for i in range(0, k):
            pred_label_temp = sortlist[i][0]
            #print 'WoW'
            #print pred_label_temp
            #print input_query_label
            if input_query_label==pred_label_temp:
                num += 1.0
            num_temp = num/((i+1)*1.0)
            num_p = num_p + num_temp

        ap += num_p / num
        #print 'NUM'
        #print num
        #num /= (k * 1.0)

        #p += num
        #cnt += 1

    return ap/query_size#np.mean(ap)

'''
def precision_at_k_user(sess, model, query_pos_test, query_pos_train, query_url_feature, k=5):
    p_list = []
    query_test_list = sorted(query_pos_test.keys())
    for query in query_test_list:
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))
        if len(pred_list) < k:
            continue

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: pred_list_feature})
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        num = 0.0
        for i in range(0, k):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                num += 1.0
        num /= (k * 1.0)

        p_list.append(num)

    return p_list
'''
