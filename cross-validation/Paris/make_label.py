if __name__=='__main__':

    f = open("lab/label原.txt", 'r')
    label = f.readlines()
    #print(label)
    #print(len(label))
    f.close()

    label_pos = {}
    for num in range(0,11):
        f = open("lab/{0}".format(num), 'r')
        label_pos[num] = f.readlines()
        #print(label_pos[num])
        f.close()
    num=0
    f = open("lab/label制作.txt","w+")
    for i in range(0,len(label)):
        for j in range(0,11):
            for k in range(0,len(label_pos[j])):
                if label[i] == label_pos[j][k]:
                    f.write(str(j) + "." + str(label[i]))
                    num=1
                    break
            if num == 1:
                break
        print(i)
        if num == 0:
            f.write(str(11) + "." + str(label[i]))
        num = 0
    f.close()