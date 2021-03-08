def getlab(strlab):
    index_label = strlab.index(".")
    return strlab[index_label+1:]
if __name__=='__main__':

    f = open("image_dataset_features_Oxford_R_mac+RA.txt", 'r')
    feature = f.readlines()
    #print(feature)
    #print(len(feature))
    f.close()

    f = open("label_dataset_features_Oxford_R_mac+RA.txt", 'r')
    label = f.readlines()
    #print(label)
    #print(len(label))
    f.close()

    f = open("lab/pos.txt", 'r')
    label_pos = f.readlines()
    #print(label_pos)
    #print(len(label_pos))
    f.close()

    f = open("lab/query.txt", 'r')
    label_query = f.readlines()
    #print(label_query)
    #print(len(label_query))
    f.close()

    f1 = open("train_feature.txt","w+")
    f2 = open("train_label.txt", "w+")
    #f3 = open("test_feature.txt","w+")
    #f4 = open("test_label.txt","w+")

    for i in range(0,len(label)):
        for j in range(len(label_pos)):
            #print(getlab(label[i]))
            #print(label_pos[j])
            if getlab(label[i]) == label_pos[j]:
                f1.write(str(feature[i]))
                f2.write(str(label[i]))
                break
        print(i)
    """
    for i in range(0, len(label)):
        for j in range(len(label_query)):
            #print(getlab(label[i]))
            #print(label_query[j])
            if getlab(label[i]) == label_query[j]:
                f3.write(str(feature[i]))
                f4.write(str(label[i]))
                break
        print(i)
    """
    f1.close()
    f2.close()
    #f3.close()
    #f4.close()