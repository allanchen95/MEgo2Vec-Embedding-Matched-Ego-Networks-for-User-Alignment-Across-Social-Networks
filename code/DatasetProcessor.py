import numpy as np
from functools import cmp_to_key
import re
from Utils import *


class DatasetProcessor(object):

    def __init__(self,name,aff,edu,pub,word_voca,char_voca, pairs,labeleddata, max_neighbor_size,batch_size=64):

        #self.labels = labeleddata[:][1]
        self.labels=np.asarray(map(lambda x: x[1], labeleddata))
        #print self.labels
        self.batch_size = batch_size
        data_size = len(labeleddata)
        self.batch_count = int(data_size / batch_size)
        #print self.batch_count

        if data_size % batch_size != 0:
            self.batch_count += 1

        self.name = name
        self.aff = aff
        self.edu = edu
        self.pub = pub
        self.word_voca = word_voca
        self.char_voca = char_voca

        self.input1_wname = []
        self.input_wname_len1 = []
        self.input1_waff = []
        self.input_waff_len1 = []
        self.input1_wedu = []
        self.input_wedu_len1 = []
        self.input1_wpub = []
        self.input_wpub_len1 = []

        self.input1_name = []
        self.input1_name_maxlen = []
        self.input1_aff = []
        self.input1_aff_maxlen = []
        self.input1_edu = []
        self.input1_edu_maxlen = []
        self.input1_pub = []
        self.input1_pub_maxlen = []

        self.neighbor_wname1 = []
        self.neighbor_wname_len1 = []
        self.neighbor_waff1 = []
        self.neighbor_waff_len1 = []
        self.neighbor_wedu1 = []
        self.neighbor_wedu_len1 = []
        self.neighbor_wpub1 = []
        self.neighbor_wpub_len1 = []

        self.neighbor_name1 = []
        self.neighbor_name1_maxlen = []
        self.neighbor_aff1 = []
        self.neighbor_aff1_maxlen = []
        self.neighbor_edu1 = []
        self.neighbor_edu1_maxlen = []
        self.neighbor_pub1 = []
        self.neighbor_pub1_maxlen = []

        self.input2_wname = []
        self.input_wname_len2 = []
        self.input2_waff = []
        self.input_waff_len2 = []
        self.input2_wedu = []
        self.input_wedu_len2 = []
        self.input2_wpub = []
        self.input_wpub_len2 = []

        self.input2_name = []
        self.input2_name_maxlen = []
        self.input2_aff = []
        self.input2_aff_maxlen = []
        self.input2_edu = []
        self.input2_edu_maxlen = []
        self.input2_pub = []
        self.input2_pub_maxlen = []

        self.neighbor_wname2 = []
        self.neighbor_wname_len2 = []
        self.neighbor_waff2 = []
        self.neighbor_waff_len2 = []
        self.neighbor_wedu2 = []
        self.neighbor_wedu_len2 = []
        self.neighbor_wpub2 = []
        self.neighbor_wpub_len2 = []

        self.neighbor_name2 = []
        self.neighbor_name2_maxlen = []
        self.neighbor_aff2 = []
        self.neighbor_aff2_maxlen = []
        self.neighbor_edu2 = []
        self.neighbor_edu2_maxlen = []
        self.neighbor_pub2 = []
        self.neighbor_pub2_maxlen = []

        self.neighbor_size = []
        self.neighbor_index = []

        self.neighbor_a=[]



        for i in range(self.batch_count):
            batch_labeleddata=labeleddata[i * batch_size:(i + 1) * batch_size]
            batch_neighbor_size = np.asarray(map(lambda x: x[2], batch_labeleddata))

            batch_x = pairs[np.asarray(map(lambda x: x[0], batch_labeleddata))]
            batch_x1 = batch_x[:, 0]
            batch_x2 = batch_x[:, 1]

            batch_neighbor_index, pad_a = self.process_network(batch_neighbor_size,batch_labeleddata, max_neighbor_size)

            size=len(batch_neighbor_index)


            flattened_batch_neighbor_index = np.concatenate(batch_neighbor_index)
            flattened_batch_neighbor_pair_ID = pairs[flattened_batch_neighbor_index]
            batch_neighbor1 = flattened_batch_neighbor_pair_ID[:, 0]
            batch_neighbor2 = flattened_batch_neighbor_pair_ID[:, 1]

            flattened_size = np.shape(batch_neighbor1)[0]

            batch_x1_wname, batch_wname_len1, \
            batch_x1_waff, batch_waff_len1, \
            batch_x1_wedu, batch_wedu_len1, \
            batch_x1_wpub, batch_wpub_len1, \
            batch_x1_name, batch_x1_name_maxlen, \
            batch_x1_aff, batch_x1_aff_maxlen, \
            batch_x1_edu, batch_x1_edu_maxlen, \
            batch_x1_pub, batch_x1_pub_maxlen,\
            batch_flattened_neighbor_wname1, batch_flattened_neighbor_wname_len1, \
            batch_flattened_neighbor_waff1, batch_flattened_neighbor_waff_len1, \
            batch_flattened_neighbor_wedu1, batch_flattened_neighbor_wedu_len1, \
            batch_flattened_neighbor_wpub1, batch_flattened_neighbor_wpub_len1,\
            batch_flattened_neighbor_name1, batch_flattened_neighbor_name1_maxlen, \
            batch_flattened_neighbor_aff1, batch_flattened_neighbor_aff1_maxlen, \
            batch_flattened_neighbor_edu1, batch_flattened_neighbor_edu1_maxlen, \
            batch_flattened_neighbor_pub1, batch_flattened_neighbor_pub1_maxlen = self.pad_feature(batch_x1,size,batch_neighbor1,flattened_size)

            batch_x2_wname, batch_wname_len2, \
            batch_x2_waff, batch_waff_len2, \
            batch_x2_wedu, batch_wedu_len2, \
            batch_x2_wpub, batch_wpub_len2, \
            batch_x2_name, batch_x2_name_maxlen, \
            batch_x2_aff, batch_x2_aff_maxlen, \
            batch_x2_edu, batch_x2_edu_maxlen, \
            batch_x2_pub, batch_x2_pub_maxlen,\
            batch_flattened_neighbor_wname2, batch_flattened_neighbor_wname_len2, \
            batch_flattened_neighbor_waff2, batch_flattened_neighbor_waff_len2, \
            batch_flattened_neighbor_wedu2, batch_flattened_neighbor_wedu_len2, \
            batch_flattened_neighbor_wpub2, batch_flattened_neighbor_wpub_len2,\
            batch_flattened_neighbor_name2, batch_flattened_neighbor_name2_maxlen, \
            batch_flattened_neighbor_aff2, batch_flattened_neighbor_aff2_maxlen, \
            batch_flattened_neighbor_edu2, batch_flattened_neighbor_edu2_maxlen, \
            batch_flattened_neighbor_pub2, batch_flattened_neighbor_pub2_maxlen = self.pad_feature(batch_x2,size,batch_neighbor2, flattened_size)

            self.input1_wname.append(batch_x1_wname)
            self.input_wname_len1.append(batch_wname_len1)
            self.input1_waff.append(batch_x1_waff)
            self.input_waff_len1.append(batch_waff_len1)
            self.input1_wedu.append(batch_x1_wedu)
            self.input_wedu_len1.append(batch_wedu_len1)
            self.input1_wpub.append(batch_x1_wpub)
            self.input_wpub_len1.append(batch_wpub_len1)

            self.input1_name.append(batch_x1_name)
            self.input1_name_maxlen.append(batch_x1_name_maxlen)
            self.input1_aff.append(batch_x1_aff)
            self.input1_aff_maxlen.append(batch_x1_aff_maxlen)
            self.input1_edu.append(batch_x1_edu)
            self.input1_edu_maxlen.append(batch_x1_edu_maxlen)
            self.input1_pub.append(batch_x1_pub)
            self.input1_pub_maxlen.append(batch_x1_pub_maxlen)

            self.neighbor_wname1.append(batch_flattened_neighbor_wname1)
            self.neighbor_wname_len1.append(batch_flattened_neighbor_wname_len1)
            self.neighbor_waff1.append(batch_flattened_neighbor_waff1)
            self.neighbor_waff_len1.append(batch_flattened_neighbor_waff_len1)
            self.neighbor_wedu1.append(batch_flattened_neighbor_wedu1)
            self.neighbor_wedu_len1.append(batch_flattened_neighbor_wedu_len1)
            self.neighbor_wpub1.append(batch_flattened_neighbor_wpub1)
            self.neighbor_wpub_len1.append(batch_flattened_neighbor_wpub_len1)

            self.neighbor_name1.append(batch_flattened_neighbor_name1)
            self.neighbor_name1_maxlen.append(batch_flattened_neighbor_name1_maxlen)
            self.neighbor_aff1.append(batch_flattened_neighbor_aff1)
            self.neighbor_aff1_maxlen.append(batch_flattened_neighbor_aff1_maxlen)
            self.neighbor_edu1.append(batch_flattened_neighbor_edu1)
            self.neighbor_edu1_maxlen.append(batch_flattened_neighbor_edu1_maxlen)
            self.neighbor_pub1.append(batch_flattened_neighbor_pub1)
            self.neighbor_pub1_maxlen.append(batch_flattened_neighbor_pub1_maxlen)

            self.input2_wname.append(batch_x2_wname)
            self.input_wname_len2.append(batch_wname_len2)
            self.input2_waff.append(batch_x2_waff)
            self.input_waff_len2.append(batch_waff_len2)
            self.input2_wedu.append(batch_x2_wedu)
            self.input_wedu_len2.append(batch_wedu_len2)
            self.input2_wpub.append(batch_x2_wpub)
            self.input_wpub_len2.append(batch_wpub_len2)

            self.input2_name.append(batch_x2_name)
            self.input2_name_maxlen.append(batch_x2_name_maxlen)
            self.input2_aff.append(batch_x2_aff)
            self.input2_aff_maxlen.append(batch_x2_aff_maxlen)
            self.input2_edu.append(batch_x2_edu)
            self.input2_edu_maxlen.append(batch_x2_edu_maxlen)
            self.input2_pub.append(batch_x2_pub)
            self.input2_pub_maxlen.append(batch_x2_pub_maxlen)

            self.neighbor_wname2.append(batch_flattened_neighbor_wname2)
            self.neighbor_wname_len2.append(batch_flattened_neighbor_wname_len2)
            self.neighbor_waff2.append(batch_flattened_neighbor_waff2)
            self.neighbor_waff_len2.append(batch_flattened_neighbor_waff_len2)
            self.neighbor_wedu2.append(batch_flattened_neighbor_wedu2)
            self.neighbor_wedu_len2.append(batch_flattened_neighbor_wedu_len2)
            self.neighbor_wpub2.append(batch_flattened_neighbor_wpub2)
            self.neighbor_wpub_len2.append(batch_flattened_neighbor_wpub_len2)

            self.neighbor_name2.append(batch_flattened_neighbor_name2)
            self.neighbor_name2_maxlen.append(batch_flattened_neighbor_name2_maxlen)
            self.neighbor_aff2.append(batch_flattened_neighbor_aff2)
            self.neighbor_aff2_maxlen.append(batch_flattened_neighbor_aff2_maxlen)
            self.neighbor_edu2.append(batch_flattened_neighbor_edu2)
            self.neighbor_edu2_maxlen.append(batch_flattened_neighbor_edu2_maxlen)
            self.neighbor_pub2.append(batch_flattened_neighbor_pub2)
            self.neighbor_pub2_maxlen.append(batch_flattened_neighbor_pub2_maxlen)

            self.neighbor_size.append(batch_neighbor_size)
            self.neighbor_index.append(np.reshape(np.concatenate(batch_neighbor_index, axis=0), [size, max_neighbor_size]))
            self.neighbor_a.append(pad_a)


    def process_network(self, batch_neighbor_size, batch_labeleddata, max_neighbor_size):
        batch_neighbor_index = []
        batch_neighbor_a=[]
        for u,v in enumerate(batch_neighbor_size):

            if(v-1==0):
                batch_neighbor_index.append([batch_labeleddata[u][0]])
                batch_neighbor_a.append([1])
            else:
                batch_neighbor_index.append(np.concatenate(([batch_labeleddata[u][0]],batch_labeleddata[u][3:v+2]),axis=0))
                batch_neighbor_a.append(batch_labeleddata[u][v+2:])
        size=len(batch_neighbor_index)

        pad_a = np.zeros([size, max_neighbor_size, max_neighbor_size], dtype=np.int32)

        for u in range(size):
            len_a = batch_neighbor_size[u]
            index_a = np.reshape(batch_neighbor_a[u], [len_a, len_a])

            a_sum = np.sum(index_a, axis=0)
            a_sort = np.argsort(-a_sum)
            a_normal = np.zeros([len_a, len_a], dtype=np.int32)
            for v in range(len_a):
                for z in range(len_a):
                    a_normal[v][z]=index_a[a_sort[v]][a_sort[z]]


            pad_a[u][0:0 + index_a.shape[0], 0:0 + index_a.shape[1]] = a_normal

            batch_neighbor_index[u]=np.asarray(batch_neighbor_index[u])[a_sort]


        for j in range(size):
            batch_neighbor_index[j] = np.concatenate((batch_neighbor_index[j],np.zeros(max_neighbor_size - batch_neighbor_size[j],dtype=np.int32))).astype(np.int32)
        return batch_neighbor_index, pad_a

    def process_features(self, raw_features, vocabulary):

        features = []
        feature_length = []
        featureIDs = []

        text = raw_features
        #text = map(lambda line: re.split('[\s\-\(\)"\[\]]+', line.strip().lower()), text)
        feature = list(map(lambda line: list(filter(lambda word_id: word_id != -1, map(
            lambda word: vocabulary[word], line))), text))
        size = len(text)
        feature_length = np.reshape(list(map(lambda x: len(x), feature)), [size, 1])
        for i in xrange(size):
            sample=[]
            sample = np.concatenate((sample, feature[i])).astype(np.int32)
            features.append(sample)
        return np.asarray(features), feature_length

    def process_char(self, raw_features, vocabulary):

        max_length=0
        charids=[]
        character=[]
        char_feature=[]
        index_sentence=[]
        text = raw_features
        #text = map(lambda line: re.split('[\s\-\(\)"\[\]]+', line.strip().lower()), text)
        for j in text:
            chars = list(map(lambda line: list(line), j))
            for char in chars:
                if (len(char)>max_length):
                    max_length=len(char)
                for c in char:
                    try:
                        if(vocabulary[c]!=-1):
                            charids.append(vocabulary[c])
                    except KeyError:
                        print 'key error:%s'%(c)
                if(charids!=[]):
                    char_feature.append(charids)
                charids = []
            index_sentence.append(char_feature)
            char_feature=[]

        return np.asarray(index_sentence), max_length

    #pad char
    def pad_char(self, index_sentences,max_sent_length, max_char_per_word):
        c = np.zeros([len(index_sentences), max_sent_length, max_char_per_word], dtype=np.int32)
        # this is to mark space at the end of the words
        word_end_id=0

        for i in range(len(index_sentences)):
            words = index_sentences[i]
            #print words
            sent_length = len(words)
            for j in range(min(sent_length, max_sent_length)):
                chars = words[j]
                char_length = len(chars)
                for k in range(min(char_length, max_char_per_word)):
                    cid = chars[k]
                    c[i, j, k] = cid
                # fill index of word end after the end of word
                #c[i, j, char_length:] = word_end_id
            # Zero out C after the end of the sentence
            #c[i, sent_length:, :] = 0
            #print c
        return c

        # pad zero for feature
    def pad_feature(self, index, size,neighbor_index,f_size):

        #process word-level
        x_wname = self.name[index]
        x_waff = self.aff[index]
        x_wedu = self.edu[index]
        x_wpub = self.pub[index]

        index_wname, wname_len = self.process_features(x_wname, self.word_voca)
        index_waff, waff_len = self.process_features(x_waff, self.word_voca)
        index_wedu, wedu_len = self.process_features(x_wedu, self.word_voca)
        index_wpub, wpub_len = self.process_features(x_wpub, self.word_voca)

        max_wname_len = max(np.sum(wname_len, axis=1))
        max_waff_len = max(np.sum(waff_len, axis=1))
        max_wedu_len = max(np.sum(wedu_len, axis=1))
        max_wpub_len = max(np.sum(wpub_len, axis=1))

        for j in range(size):
            index_wname[j] = np.concatenate((index_wname[j], np.zeros(max_wname_len - wname_len[j], dtype=np.int32)))
            index_waff[j] = np.concatenate((index_waff[j], np.zeros(max_waff_len - waff_len[j], dtype=np.int32)))
            index_wedu[j] = np.concatenate((index_wedu[j], np.zeros(max_wedu_len - wedu_len[j], dtype=np.int32)))
            index_wpub[j] = np.concatenate((index_wpub[j], np.zeros(max_wpub_len - wpub_len[j], dtype=np.int32)))

        index_wname = np.reshape(np.concatenate(index_wname, axis=0), [size, max_wname_len])
        index_waff = np.reshape(np.concatenate(index_waff, axis=0), [size, max_waff_len])
        index_wedu = np.reshape(np.concatenate(index_wedu, axis=0), [size, max_wedu_len])
        index_wpub = np.reshape(np.concatenate(index_wpub, axis=0), [size, max_wpub_len])

        # process char-level

        index_cname, cname_maxlen = self.process_char(x_wname, self.char_voca)
        index_caff, caff_maxlen = self.process_char(x_waff, self.char_voca)
        index_cedu, cedu_maxlen = self.process_char(x_wedu, self.char_voca)
        index_cpub, cpub_maxlen = self.process_char(x_wpub, self.char_voca)


        index_cname = self.pad_char(index_cname, max_wname_len, cname_maxlen)
        index_caff = self.pad_char(index_caff, max_waff_len, caff_maxlen)
        index_cedu = self.pad_char(index_cedu, max_wedu_len, cedu_maxlen)
        index_cpub = self.pad_char(index_cpub, max_wpub_len, cpub_maxlen)



        index_cname = np.reshape(np.concatenate(index_cname, axis=0), [size, max_wname_len, cname_maxlen])
        index_caff = np.reshape(np.concatenate(index_caff, axis=0), [size, max_waff_len, caff_maxlen])
        index_cedu = np.reshape(np.concatenate(index_cedu, axis=0), [size, max_wedu_len, cedu_maxlen])
        index_cpub = np.reshape(np.concatenate(index_cpub, axis=0), [size, max_wpub_len, cpub_maxlen])

        #process neighbor word level
        flattened_neighbor_wname = self.name[neighbor_index]
        flattened_neighbor_waff = self.aff[neighbor_index]
        flattened_neighbor_wedu = self.edu[neighbor_index]
        flattened_neighbor_wpub = self.pub[neighbor_index]

        index_flattened_neighbor_wname, flattened_neighbor_wname_len = self.process_features(flattened_neighbor_wname, self.word_voca)
        index_flattened_neighbor_waff, flattened_neighbor_waff_len = self.process_features(flattened_neighbor_waff, self.word_voca)
        index_flattened_neighbor_wedu, flattened_neighbor_wedu_len = self.process_features(flattened_neighbor_wedu, self.word_voca)
        index_flattened_neighbor_wpub, flattened_neighbor_wpub_len = self.process_features(flattened_neighbor_wpub, self.word_voca)

        max_neighbor_wname_length = max(np.sum(flattened_neighbor_wname_len,axis=1))
        max_neighbor_waff_length = max(np.sum(flattened_neighbor_waff_len,axis=1))
        max_neighbor_wedu_length = max(np.sum(flattened_neighbor_wedu_len,axis=1))
        max_neighbor_wpub_length = max(np.sum(flattened_neighbor_wpub_len,axis=1))

        for j in range(f_size):
            index_flattened_neighbor_wname[j] = np.concatenate((index_flattened_neighbor_wname[j], np.zeros(max_neighbor_wname_length - flattened_neighbor_wname_len[j], dtype=np.int32)))
            index_flattened_neighbor_waff[j] = np.concatenate((index_flattened_neighbor_waff[j], np.zeros(max_neighbor_waff_length - flattened_neighbor_waff_len[j], dtype=np.int32)))
            index_flattened_neighbor_wedu[j] = np.concatenate((index_flattened_neighbor_wedu[j], np.zeros(max_neighbor_wedu_length - flattened_neighbor_wedu_len[j], dtype=np.int32)))
            index_flattened_neighbor_wpub[j] = np.concatenate((index_flattened_neighbor_wpub[j], np.zeros(max_neighbor_wpub_length - flattened_neighbor_wpub_len[j], dtype=np.int32)))

        index_flattened_neighbor_wname = np.reshape(np.concatenate(index_flattened_neighbor_wname, axis=0), [f_size, max_neighbor_wname_length])
        index_flattened_neighbor_waff = np.reshape(np.concatenate(index_flattened_neighbor_waff, axis=0), [f_size, max_neighbor_waff_length])
        index_flattened_neighbor_wedu = np.reshape(np.concatenate(index_flattened_neighbor_wedu, axis=0), [f_size, max_neighbor_wedu_length])
        index_flattened_neighbor_wpub = np.reshape(np.concatenate(index_flattened_neighbor_wpub, axis=0), [f_size, max_neighbor_wpub_length])

        index_flattened_neighbor_cname, flattened_neighbor_cname_maxlen = self.process_char(flattened_neighbor_wname, self.char_voca)
        index_flattened_neighbor_caff, flattened_neighbor_caff_maxlen = self.process_char(flattened_neighbor_waff, self.char_voca)
        index_flattened_neighbor_cedu, flattened_neighbor_cedu_maxlen = self.process_char(flattened_neighbor_wedu, self.char_voca)
        index_flattened_neighbor_cpub, flattened_neighbor_cpub_maxlen = self.process_char(flattened_neighbor_wpub, self.char_voca)


        index_flattened_neighbor_cname = self.pad_char(index_flattened_neighbor_cname, max_neighbor_wname_length, flattened_neighbor_cname_maxlen)
        index_flattened_neighbor_caff = self.pad_char(index_flattened_neighbor_caff, max_neighbor_waff_length,flattened_neighbor_caff_maxlen)
        index_flattened_neighbor_cedu = self.pad_char(index_flattened_neighbor_cedu, max_neighbor_wedu_length, flattened_neighbor_cedu_maxlen)
        index_flattened_neighbor_cpub = self.pad_char(index_flattened_neighbor_cpub, max_neighbor_wpub_length, flattened_neighbor_cpub_maxlen)



        index_flattened_neighbor_cname = np.reshape(np.concatenate(index_flattened_neighbor_cname, axis=0), [f_size, max_neighbor_wname_length, flattened_neighbor_cname_maxlen])
        index_flattened_neighbor_caff = np.reshape(np.concatenate(index_flattened_neighbor_caff, axis=0), [f_size, max_neighbor_waff_length, flattened_neighbor_caff_maxlen])
        index_flattened_neighbor_cedu = np.reshape(np.concatenate(index_flattened_neighbor_cedu, axis=0), [f_size, max_neighbor_wedu_length, flattened_neighbor_cedu_maxlen])
        index_flattened_neighbor_cpub = np.reshape(np.concatenate(index_flattened_neighbor_cpub, axis=0), [f_size, max_neighbor_wpub_length, flattened_neighbor_cpub_maxlen])



        return index_wname,wname_len,index_waff,waff_len,index_wedu,wedu_len,index_wpub,wpub_len,\
               index_cname,cname_maxlen,index_caff,caff_maxlen,index_cedu,cedu_maxlen,index_cpub,cpub_maxlen, \
               index_flattened_neighbor_wname, flattened_neighbor_wname_len, \
               index_flattened_neighbor_waff, flattened_neighbor_waff_len, \
               index_flattened_neighbor_wedu, flattened_neighbor_wedu_len, \
               index_flattened_neighbor_wpub, flattened_neighbor_wpub_len, \
               index_flattened_neighbor_cname, flattened_neighbor_cname_maxlen, \
               index_flattened_neighbor_caff, flattened_neighbor_caff_maxlen, \
               index_flattened_neighbor_cedu, flattened_neighbor_cedu_maxlen, \
               index_flattened_neighbor_cpub, flattened_neighbor_cpub_maxlen



    def get_batch(self, i):
        batch_wname1 = self.input1_wname[i]
        batch_wname_len1 = self.input_wname_len1[i]
        batch_waff1 = self.input1_waff[i]
        batch_waff_len1 = self.input_waff_len1[i]
        batch_wedu1 = self.input1_wedu[i]
        batch_wedu_len1 = self.input_wedu_len1[i]
        batch_wpub1 = self.input1_wpub[i]
        batch_wpub_len1 = self.input_wpub_len1[i]

        batch_name1 = self.input1_name[i]
        batch_name1_maxlen = self.input1_name_maxlen[i]
        batch_aff1 = self.input1_aff[i]
        batch_aff1_maxlen = self.input1_aff_maxlen[i]
        batch_edu1 = self.input1_edu[i]
        batch_edu1_maxlen = self.input1_edu_maxlen[i]
        batch_pub1 = self.input1_pub[i]
        batch_pub1_maxlen = self.input1_pub_maxlen[i]

        batch_neighbor_wname1 = self.neighbor_wname1[i]
        batch_neighbor_wname_len1 = self.neighbor_wname_len1[i]
        batch_neighbor_waff1 = self.neighbor_waff1[i]
        batch_neighbor_waff_len1 = self.neighbor_waff_len1[i]
        batch_neighbor_wedu1 = self.neighbor_wedu1[i]
        batch_neighbor_wedu_len1 = self.neighbor_wedu_len1[i]
        batch_neighbor_wpub1 = self.neighbor_wpub1[i]
        batch_neighbor_wpub_len1 = self.neighbor_wpub_len1[i]

        batch_neighbor_name1 = self.neighbor_name1[i]
        batch_neighbor_name1_maxlen = self.neighbor_name1_maxlen[i]
        batch_neighbor_aff1 = self.neighbor_aff1[i]
        batch_neighbor_aff1_maxlen = self.neighbor_aff1_maxlen[i]
        batch_neighbor_edu1 = self.neighbor_edu1[i]
        batch_neighbor_edu1_maxlen = self.neighbor_edu1_maxlen[i]
        batch_neighbor_pub1 = self.neighbor_pub1[i]
        batch_neighbor_pub1_maxlen = self.neighbor_pub1_maxlen[i]

        batch_wname2 = self.input2_wname[i]
        batch_wname_len2 = self.input_wname_len2[i]
        batch_waff2 = self.input2_waff[i]
        batch_waff_len2 = self.input_waff_len2[i]
        batch_wedu2 = self.input2_wedu[i]
        batch_wedu_len2 = self.input_wedu_len2[i]
        batch_wpub2 = self.input2_wpub[i]
        batch_wpub_len2 = self.input_wpub_len2[i]

        batch_name2 = self.input2_name[i]
        batch_name2_maxlen = self.input2_name_maxlen[i]
        batch_aff2 = self.input2_aff[i]
        batch_aff2_maxlen = self.input2_aff_maxlen[i]
        batch_edu2 = self.input2_edu[i]
        batch_edu2_maxlen = self.input2_edu_maxlen[i]
        batch_pub2 = self.input2_pub[i]
        batch_pub2_maxlen = self.input2_pub_maxlen[i]

        batch_neighbor_wname2 = self.neighbor_wname2[i]
        batch_neighbor_wname_len2 = self.neighbor_wname_len2[i]
        batch_neighbor_waff2 = self.neighbor_waff2[i]
        batch_neighbor_waff_len2 = self.neighbor_waff_len2[i]
        batch_neighbor_wedu2 = self.neighbor_wedu2[i]
        batch_neighbor_wedu_len2 = self.neighbor_wedu_len2[i]
        batch_neighbor_wpub2 = self.neighbor_wpub2[i]
        batch_neighbor_wpub_len2 = self.neighbor_wpub_len2[i]

        batch_neighbor_name2 = self.neighbor_name2[i]
        batch_neighbor_name2_maxlen = self.neighbor_name2_maxlen[i]
        batch_neighbor_aff2 = self.neighbor_aff2[i]
        batch_neighbor_aff2_maxlen = self.neighbor_aff2_maxlen[i]
        batch_neighbor_edu2 = self.neighbor_edu2[i]
        batch_neighbor_edu2_maxlen = self.neighbor_edu2_maxlen[i]
        batch_neighbor_pub2 = self.neighbor_pub2[i]
        batch_neighbor_pub2_maxlen = self.neighbor_pub2_maxlen[i]

        batch_neighbor_size = self.neighbor_size[i]
        batch_neighbor_index = self.neighbor_index[i]
        batch_neighbor_a=self.neighbor_a[i]
        batch_y = self.labels[i * self.batch_size:(i + 1) * self.batch_size]




        return  batch_wname1,batch_wname_len1,\
                batch_waff1,batch_waff_len1,\
                batch_wedu1,batch_wedu_len1,\
                batch_wpub1,batch_wpub_len1,\
                batch_name1,batch_name1_maxlen,\
                batch_aff1,batch_aff1_maxlen,\
                batch_edu1,batch_edu1_maxlen,\
                batch_pub1,batch_pub1_maxlen,\
                batch_neighbor_wname1,batch_neighbor_wname_len1,\
                batch_neighbor_waff1,batch_neighbor_waff_len1,\
                batch_neighbor_wedu1,batch_neighbor_wedu_len1,\
                batch_neighbor_wpub1,batch_neighbor_wpub_len1,\
                batch_neighbor_name1,batch_neighbor_name1_maxlen,\
                batch_neighbor_aff1,batch_neighbor_aff1_maxlen,\
                batch_neighbor_edu1,batch_neighbor_edu1_maxlen,\
                batch_neighbor_pub1,batch_neighbor_pub1_maxlen, \
                batch_wname2, batch_wname_len2, \
                batch_waff2, batch_waff_len2, \
                batch_wedu2, batch_wedu_len2, \
                batch_wpub2, batch_wpub_len2, \
                batch_name2, batch_name2_maxlen, \
                batch_aff2, batch_aff2_maxlen, \
                batch_edu2, batch_edu2_maxlen, \
                batch_pub2, batch_pub2_maxlen, \
                batch_neighbor_wname2, batch_neighbor_wname_len2, \
                batch_neighbor_waff2, batch_neighbor_waff_len2, \
                batch_neighbor_wedu2, batch_neighbor_wedu_len2, \
                batch_neighbor_wpub2, batch_neighbor_wpub_len2, \
                batch_neighbor_name2, batch_neighbor_name2_maxlen, \
                batch_neighbor_aff2, batch_neighbor_aff2_maxlen, \
                batch_neighbor_edu2, batch_neighbor_edu2_maxlen, \
                batch_neighbor_pub2, batch_neighbor_pub2_maxlen, \
                batch_neighbor_size, batch_neighbor_index,\
                batch_neighbor_a,batch_y
