import numpy as np
import json
import random
import gzip
import math
import collections as colls
import copy

class upsearch_data:
    def __init__(self, data_path, input_train_dir,
            set_name, batch_size, model_struct,
            threshold='strict', info_level = 1,
            is_av_embed_shared = True,
            is_feedback_same_user = False,
            keep_feedback_type = 'pos_neg', #"pos", "neg"
            feedback_user_model = 'first', #ra, fa, rv, fv
            n_negs = 5,
            max_av_count = 5,
            neg_per_pos = 3):
        #get product/user/vocabulary information
        self.info_level = info_level
        self.is_feedback_same_user = is_feedback_same_user
        print "info_level:%d" % self.info_level
        self.is_av_embed_shared = is_av_embed_shared
        self.keep_feedback_type = keep_feedback_type
        self.feedback_user_model = feedback_user_model #first, random, aspect
        self.batch_size = batch_size
        self.neg_per_pos = neg_per_pos
        self.n_negs = n_negs
        self.model_struct = model_struct
        self.max_av_count = max_av_count #predefine the max number of av to be 5. five aspects overlap
        self.product_ids = []
        #read products ASINID to list
        with gzip.open("%s/product.txt.gz" % data_path, 'r') as fin:
            for line in fin:
                self.product_ids.append(line.strip())
        self.product_size = len(self.product_ids)
        #read user ASINID to list
        self.user_ids = []
        with gzip.open("%s/users.txt.gz" % data_path, 'r') as fin:
            for line in fin:
                self.user_ids.append(line.strip())
        self.user_size = len(self.user_ids)
        #read aspect, values in 'av.train.relax.txt.gz' or 'av.train.strict.txt.gz'
        self.words = []
        self.not_idx = -1
        with gzip.open("%s/vocab.txt.gz" % data_path, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == "not":
                    self.not_idx = len(self.words)
                self.words.append(line)
        self.vocab_size = len(self.words)
        # index self.vocab_size is used for <S> end symbol
        self.padding_idx = self.vocab_size #0, vocab_size - 1, padding
        self.query_words = []
        self.query_max_length = 0
        with gzip.open("%s/query.txt.gz" % input_train_dir, 'r') as fin:
            for line in fin:
                words = [int(i) for i in line.strip().split(' ')]
                if len(words) > self.query_max_length:
                    self.query_max_length = len(words)
                self.query_words.append(words)
        #pad -1 before actual query words
        #different from embedding lookup function in tensorflow, -1 will cause index out of range in pytorch
        # we need to use actual padding index
        for i in xrange(len(self.query_words)):
            self.query_words[i] = [self.padding_idx for j in xrange(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]

        #get review sets
        self.word_count = 0
        self.vocab_distribute = np.zeros(self.vocab_size)
        self.review_info = []
        self.review_text = []
        with gzip.open("%s/%s.txt.gz" % (input_train_dir, set_name), 'r') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                self.review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
                self.review_text.append([int(i) for i in arr[2].split(' ')])
                for idx in self.review_text[-1]:
                    self.vocab_distribute[idx] += 1
                self.word_count += len(self.review_text[-1])
        self.review_size = len(self.review_info)
        self.vocab_distribute = self.vocab_distribute.tolist()
        self.sub_sampling_rate = None
        #self.review_distribute = np.ones(self.review_size).tolist()
        self.product_distribute = np.ones(self.product_size).tolist()
        self.word_dists = self.neg_distributes(self.vocab_distribute)
        self.product_dists = self.neg_distributes(self.product_distribute)

        print "total_word_count:%d" % self.word_count

        #get product query sets
        self.product_query_idx = []
        with gzip.open("%s/%s_query_idx.txt.gz" % (input_train_dir, set_name), 'r') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                query_idx = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    query_idx.append(int(idx))
                self.product_query_idx.append(query_idx)

        #read av.train.txt.gz
        print("Data statistic: vocab %d, user %d, product %d\n" % (self.vocab_size,
                    self.user_size, self.product_size))
        self.value_dists = []
        self.aspect_dists = []
        self.aspect_value_count_dic = dict()
        self.av_word2id = dict() #av word id in the original vocab (reviews), in the separate vocab
        self.av_id2word = list() #id in the separate vocab, id in the original vocab (reviews)
        self.aword_idx2_aid = dict()
        self.aspect_keys = []
        self.value_keys = []
        if self.model_struct == 'AVHEM':
            self.read_overall_aspect_values(data_path, threshold)
            #print('self.av_word2id:%s' % self.av_word2id)
            #print('self.av_id2word:%s' % self.av_id2word)
            #self.read_av_of_reviews(input_train_dir, set_name, threshold)
            if set_name == 'test':
                self.test_product_av_dic = dict() # product:user:aspect tuple, value tuple
                self.read_av_of_reviews(self.test_product_av_dic, input_train_dir, 'test', threshold)
            #read av from reviews of training data, get aspect_keys, value_keys and distribution
            self.product_av_dic = dict() # product:user:aspect tuple, value tuple
            self.read_av_of_reviews(self.product_av_dic, input_train_dir, 'train', threshold)
            #overwrite self.value_count_dic and self.aspect_value_count_dic when they are from test data
            self.construct_aspect_value_keys_from_train()
            if set_name == 'test':
                self.product_av_dic = self.test_product_av_dic
            #print("word_id:%s" % self.aword_idx2_aid)
            #print("value_id:%s" % self.vword_idx2_vid)
            self.value_dists = self.neg_distributes(self.value_distr)
            self.aspect_dists = self.neg_distributes(self.aspect_distr)

    def read_overall_aspect_values(self, data_path, threshold):
        # 12499 13323|3|15:12483|12,12902|2,7958|1
        # get aspects.
        # get values of each aspect
        #self.padded_aspect_dic = dict() #original aspect: padded aspect

        self.av_word2id = dict() #av word id in the original vocab (reviews), in the separate vocab
        self.av_id2word = list() #id in the separate vocab, id in the original vocab (reviews)

        self.av_word2id[self.not_idx] = len(self.av_id2word)
        self.av_id2word.append(self.not_idx)

        with file("%s/lexicon.%s.stemmed.id.sorted.txt" % (data_path, threshold)) as fin:
            for line in fin:
                aspect_info, value_info = line.strip('\r\n').split(':')
                aspect = tuple(map(int, aspect_info.split('|')[0].split(' '))) # aspect|#value|#appeared_in_all
                values = [x.split('|')[0] for x in value_info.split(',')]
                for word in aspect:
                    if word not in self.av_word2id:
                        self.av_word2id[word] = len(self.av_id2word)
                        self.av_id2word.append(word)

                for val_str in values:
                    val = tuple(map(int, val_str.split(' ')))
                    for word in val:
                        if word not in self.av_word2id:
                            self.av_word2id[word] = len(self.av_id2word)
                            self.av_id2word.append(word)
        if self.is_av_embed_shared:
            self.av_padding_idx = self.padding_idx
        else:
            # av_padding_idx can be same as the other words or not the same
            self.av_vocab_size = len(self.av_word2id)
            self.av_padding_idx = self.av_vocab_size

    def read_av_of_reviews(self, product_av_dic, input_train_dir, set_name, threshold):
        #self.vword_idx2_vid = None # single word as value, value_id
        #self.aword_idx2_aid = None
        self.value_count_dic = colls.defaultdict(float) #single word as value, #appear
        self.aspect_value_count_dic = colls.defaultdict(dict)
        #score can be used or not
        self.max_a_length = 0
        self.max_v_length = 0
        with gzip.open("%s/av.%s.%s.txt.gz" % (input_train_dir, set_name, threshold), 'r') as fin:
            for line in fin:
                up, avs = line.strip().split(",")
                u, p = [int(x) for x in up.split("@")]
                av = dict()
                for x in avs.split(":"):
                    a = map(int, x.split('|')[0].split(' '))
                    v = map(int, x.split('|')[1].split(' '))
                    if not self.is_av_embed_shared:
                        a = [self.av_word2id[x] for x in a]
                        v = [self.av_word2id[x] for x in v]
                    a = tuple(a)
                    v = tuple(v)

                    self.max_v_length = max(self.max_v_length, len(v))
                    if len(v) > 2:
                        v = v[:2]
                    if len(v) > 1:
                        if not v[0] == self.av_word2id[self.not_idx]:
                            #if value contains two words or more and the first one is not "not", discard it
                            continue
                        else:
                            av[a] = v#only keep not and one word
                            v = v[1]
                    else:
                        av[a] = v # v is a tuple
                        v = v[0] #v becomes an element
                    #may be also tuple for v
                    self.max_a_length = max(self.max_a_length, len(a))
                    self.value_count_dic[v] += 1
                    if v not in self.aspect_value_count_dic[a]:
                        self.aspect_value_count_dic[a][v] = 0.
                    self.aspect_value_count_dic[a][v] += 1.

                if p not in product_av_dic:
                    product_av_dic[p] = dict()
                product_av_dic[p][u] = av
                #key(u,p): [aspect_wordid_str, value]

    def construct_aspect_value_keys_from_train(self):
        self.aspect_keys = self.aspect_value_count_dic.keys()
        self.aspect_distr = [sum([self.aspect_value_count_dic[x][y] \
            for y in self.aspect_value_count_dic[x]]) \
            for x in self.aspect_keys]
        av_pair_count = sum([len(self.aspect_value_count_dic[x]) \
            for x in self.aspect_keys])
        print("aspect_value_count:%s" % av_pair_count)
        #print(self.aspect_distr)
        self.aword_idx2_aid = dict() #before padding
        for a_id in xrange(len(self.aspect_keys)):
            self.aword_idx2_aid[self.aspect_keys[a_id]] = a_id

        self.value_keys = self.value_count_dic.keys()
        self.value_distr = [self.value_count_dic[x] for x in self.value_keys]
        self.aspect_keys_len = np.asarray([len(x) for x in self.aspect_keys])
        self.vword_idx2_vid = dict()
        for v_id in xrange(len(self.value_keys)):
            self.vword_idx2_vid[self.value_keys[v_id]] = v_id

        for i in xrange(len(self.aspect_keys)):
            self.aspect_keys[i] = list(self.aspect_keys[i]) \
                    + [self.av_padding_idx] * (self.max_a_length - len(self.aspect_keys[i]))
        self.aspect_keys.append([self.av_padding_idx] * self.max_a_length)
        self.aspect_keys = np.asarray(self.aspect_keys)
        self.aspect_distr.append(0.)
        self.value_keys.append(self.av_padding_idx)
        self.value_distr.append(0.)
        self.value_keys = np.asarray(self.value_keys)
        '''
        print("self.product_av_dic:%s" % self.product_av_dic)
        print("self.aspect_keys:%s" % self.aspect_keys)
        print("self.value_keys:%s" % self.value_keys)
        print("self.aspect_distr:%s" % self.aspect_distr)
        print("self.vword_idx2_vid:%s" % self.vword_idx2_vid)
        '''

        print("value_keys_count:%s" % len(self.value_keys))
        print("aspect_count: %s" % len(self.aspect_distr))

        # we need to do as array after padding,
        #otherwise, + in ndarray means numeric operator, not concatenate two lists.

    def sub_sampling(self, subsample_threshold):
        if subsample_threshold == 0.0:
            return
        self.sub_sampling_rate = [1.0 for _ in xrange(self.vocab_size)]
        threshold = sum(self.vocab_distribute) * subsample_threshold
        print "threshold:%f" % threshold
        count_sub_sample = 0
        for i in xrange(self.vocab_size):
            #vocab_distribute[i] could be zero if the word does not appear in the training set
            if self.vocab_distribute[i] == 0:
                self.sub_sampling_rate[i] = 0
                #if this word does not appear in training set, set the rate to 0.
                continue
            self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
                                            1.0)
            count_sub_sample += 1

        self.sample_count = sum([self.sub_sampling_rate[i] * self.vocab_distribute[i] for i in xrange(self.vocab_size)])
        print "sample_count", self.sample_count

    def get_av_count_dic_for_u_plist(self, product_list, uid=None):
        av_dic = colls.defaultdict(dict)
        #aspect_id (unique identifier for aspect), values (tuples)
        for p in product_list:
            if p not in self.product_av_dic:
                continue
            if uid is not None:
                if uid not in self.product_av_dic[p]:
                    continue
                for a in self.product_av_dic[p][uid]:
                    if a not in self.aword_idx2_aid:
                        continue
                    v = self.product_av_dic[p][uid][a]
                    v_nosign = v[-1]
                    if v_nosign not in self.vword_idx2_vid:
                        #value in test set may not appear in training set, based on which vword_idx2_id is built
                        #print("v:")
                        #print(v)
                        continue
                    if v not in av_dic[self.aword_idx2_aid[a]]:
                        av_dic[self.aword_idx2_aid[a]][v] = 0
                    av_dic[self.aword_idx2_aid[a]][v] += 1

            else:
                for each_u in self.product_av_dic[p]:
                    for a in self.product_av_dic[p][each_u]:
                        if a not in self.aword_idx2_aid:
                            #print("a:")
                            #print(a)
                            continue
                        v = self.product_av_dic[p][each_u][a]
                        v_nosign = v[-1]
                        if v_nosign not in self.vword_idx2_vid:
                            #print("v:")
                            #print(v)
                            continue
                        if v not in av_dic[self.aword_idx2_aid[a]]:
                            av_dic[self.aword_idx2_aid[a]][v] = 0
                        av_dic[self.aword_idx2_aid[a]][v] += 1

        #if len(av_dic) == 0:
        #    return []
        return av_dic


    def get_av_dic_for_u_plist(self, product_list, uid=None):
        print uid
        av_dic = colls.defaultdict(set)
        #aspect_id (unique identifier for aspect), values (tuples)
        for p in product_list:
            if p not in self.product_av_dic:
                continue
            if uid is not None:
                if uid not in self.product_av_dic[p]:
                    continue
                for a in self.product_av_dic[p][uid]:
                    if a not in self.aword_idx2_aid:
                        continue
                    v = self.product_av_dic[p][uid][a]
                    v_nosign = v[-1]
                    if v_nosign not in self.vword_idx2_vid:
                        #value in test set may not appear in training set, based on which vword_idx2_id is built
                        #print("v:")
                        #print(v)
                        continue
                    av_dic[self.aword_idx2_aid[a]].add(v)
                    print uid, self.aword_idx2_aid[a], v
            else:
                for each_u in self.product_av_dic[p]:
                    for a in self.product_av_dic[p][each_u]:
                        if a not in self.aword_idx2_aid:
                            #print("a:")
                            #print(a)
                            continue
                        v = self.product_av_dic[p][each_u][a]
                        v_nosign = v[-1]
                        if v_nosign not in self.vword_idx2_vid:
                            #print("v:")
                            #print(v)
                            continue
                        av_dic[self.aword_idx2_aid[a]].add(v)
                        print each_u, self.aword_idx2_aid[a], v

        return av_dic

    def item_pair_to_AVs(self, posP_list, negP_list, uid=None, info_level = 1, is_test=False):
        #posP_list may just contain one element
        #if negP also contains product in posP, all the aspect values will be confirmed
        #construct AV pairs from pos-P and neg-P
        pos_av_dic = self.get_av_count_dic_for_u_plist(posP_list, uid)
        neg_av_dic = self.get_av_count_dic_for_u_plist(negP_list) #use all aspect-values from all reviews of negative products
        #aspect, set of values

        #neg av-> a1 not v1; confirm a2 is v2. level 1
        #neg av-> a1 not v1, but v2; confirm a1 is v3. level 2
        #pos av-> a1 not v1, but v2; a2 is v3 (a2 is not in neg avs, but from pos av). level 3
        #more information provided as level is up
        if not pos_av_dic:# or neg_av_dic is None:
            return []
        av_pairs = []
        for aspect in neg_av_dic:
            #aspect is an id
            has_common_value = False
            if aspect not in pos_av_dic:
                continue
            #padded_aspect = list(aspect) + [self.av_padding_idx] * (self.max_a_length - aspect_len)
            #val is tuple with 1 or 2 elements.
            for val in neg_av_dic[aspect]:
                if val not in pos_av_dic[aspect]:
                    continue
                has_common_value = True
                #if len(val) > 1, it must have 'not' in the beginning, we dicard other val in the dictionary
                if len(val) > 1 and val[0] == self.av_word2id[self.not_idx]:
                    av_pairs.append([aspect, neg_av_dic[aspect][val], self.vword_idx2_vid[val[1]], -1])
                else:
                    av_pairs.append([aspect, neg_av_dic[aspect][val], self.vword_idx2_vid[val[0]], 1])
            if info_level > 1:
                #values that are only in pos_av
                for val in pos_av_dic[aspect]:
                    if val in neg_av_dic[aspect]:
                        continue
                    if len(val) > 1 and val[0] == self.av_word2id[self.not_idx]:
                        av_pairs.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[1]], -1])
                    else:
                        av_pairs.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[0]], 1])
            else:
                #only give negative feedback
                if is_test and has_common_value:
                    #during test time, if positive value is already shown, do not include the negative value
                    continue
                for val in neg_av_dic[aspect]:
                    if val in pos_av_dic[aspect]:
                        continue
                    if len(val) > 1 and val[0] == self.av_word2id[self.not_idx]:
                       if val[1] in common_val_set:
                           # pos "good" match neg "not good"
                           av_pairs.append([aspect, neg_av_dic[aspect][val], self.vword_idx2_vid[val[1]], 1])
                        #else discard pos "great" match neg "not bad" discard

                    else:
                        av_pairs.append([aspect, neg_av_dic[aspect][val], self.vword_idx2_vid[val[0]], -1])
                #both aspect and value are tuples, since they need to be key, list cannot be used as key
        if info_level > 2:
            for aspect in pos_av_dic:
                if aspect in neg_av_dic:
                    continue
                #aspect = list(aspect) + [self.av_padding_idx] * (self.max_a_length - aspect_len)
                for val in pos_av_dic[aspect]:
                    if len(val) > 1 and self.av_word2id[self.not_idx]:
                        av_pairs.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[1]], -1])
                    else:
                        av_pairs.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[0]], 1])

        #av_pairs = [x for x in av_pairs if x[-1] == 1]
        #av_pairs.sort(key = lambda x: len(x), reverse = True)
        #sort the length according to descending order to make pack_padded_sequence,
        #since the sequences should be sorted by length in a decreasing order
        #that need is for batch level, instance level sorting doesn't help

        #this value may be larger than 5 when two items have a lot of matching aspects
        #print("av_pairs:%s" % av_pairs)
        return av_pairs

    def select_av_pairs(self, av_pairs, max_av_count):
        if self.keep_feedback_type == "pos":
            av_pairs = [x for x in av_pairs if x[-1] == 1]
        if self.keep_feedback_type == "neg":
            av_pairs = [x for x in av_pairs if x[-1] == -1]
        if len(av_pairs) <= max_av_count:
            return av_pairs
        if self.feedback_user_model == 'first':
            return av_pairs[:max_av_count]
        elif self.feedback_user_model == 'random':
            random.shuffle(av_pairs)
            return av_pairs[:max_av_count]
            #np.random.choice only take array with dim=1
        #otherwise, keep all
        aspect_count_dic = colls.defaultdict(int)
        aspect_value_dic = colls.defaultdict(list)
        aspects_indices = dict()
        for x in av_pairs:
            aspect_count_dic[x[0]] += x[1]
            aspect_value_dic[x[0]].append(x)
            aspects_indices[x[0]] = 0
        ordered_aspects = sorted(aspect_count_dic, key=aspect_count_dic.get,reverse=True)
        aspects = ordered_aspects
        if len(ordered_aspects) > max_av_count:
            if 'ra' in self.feedback_user_model: #ra,fa,
                aspects = np.random.choice(ordered_aspects, max_av_count)
            else:
                aspects = ordered_aspects[:max_av_count]
        sel_av_pairs = []
        for aspect in aspects:
            if 'rv' in self.feedback_user_model:
                random.shuffle(aspect_value_dic[aspect])
            else:
                aspect_value_dic[aspect].sort(key=lambda x:x[1], reverse=True)

        has_finish = False
        while len(sel_av_pairs) < max_av_count and not has_finish:
            has_finish = True
            for aspect in aspects:
                if aspects_indices[aspect] > len(aspect_value_dic[aspect]) - 1:
                    continue
                has_finish = False
                sel_av_pairs.append(aspect_value_dic[aspect][aspects_indices[aspect]])
                aspects_indices[aspect] += 1
                if len(sel_av_pairs) == max_av_count:
                    break

        return sel_av_pairs

    def setup_data_set(self, words_to_train):
        self.train_seq = range(len(self.review_info))
        self.words_to_train = words_to_train
        self.finished_word_num = 0

    def neg_distributes(self, weights, distortion = 0.75):
        #print weights
        weights = np.asarray(weights)
        #print weights.sum()
        wf = weights / weights.sum()
        wf = np.power(wf, distortion)
        wf = wf / wf.sum()
        return wf

    def initialize_epoch(self):
        random.shuffle(self.train_seq)
        self.review_size = len(self.train_seq)
        #a simple assumption
        #generate random numbers considering uniform distribution
        #other way can also be considered, such as random sample according to product distribution
        #according to query -> top retrieved products in last epoch
        self.estimated_entry_size = int(self.sample_count) + 1000 #estimate word count for each epoch
        self.neg_sample_products = np.random.randint(0, self.product_size, size = (self.review_size, self.neg_per_pos))
        self.u_neg_word_sample = np.random.choice(len(self.word_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.word_dists)
        self.p_neg_word_sample = np.random.choice(len(self.word_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.word_dists)
        self.u_neg_product_sample = np.random.choice(len(self.product_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.product_dists)
        if self.model_struct == 'AVHEM':
            self.u_neg_product_av_sample = np.random.choice(len(self.product_dists),
                    size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.product_dists)
            self.p_neg_aspect_sample = np.random.choice(len(self.aspect_dists),
                    size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.aspect_dists)
            self.pa_neg_value_sample = np.random.choice(len(self.value_dists),
                    size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.value_dists)
        #randint(low, high, size) high is exclusive
        self.cur_review_i = 0
        self.cur_word_i = 0
        self.cur_epoch_word_i = 0
        self.cur_entry_i = 0


    def get_train_batch(self):
        user_idxs, product_idxs, word_idxs = [],[],[]
        u_neg_words_idxs, p_neg_words_idxs, u_neg_p_idxs = [],[],[]
        query_word_idxs = []
        review_idx = self.train_seq[self.cur_review_i]
        user_idx = self.review_info[review_idx][0]
        product_idx = self.review_info[review_idx][1]
        query_idx = random.choice(self.product_query_idx[product_idx])
        text_list = self.review_text[review_idx]
        text_length = len(text_list)
        #for each product_idx, random sample some negative products (or sample according to ranking)
        #collect aspect values from the matching of product_idx with the negative products.

        while len(word_idxs) < self.batch_size:
            #print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
            #if sample this word
            if self.sub_sampling_rate == None or random.random() < self.sub_sampling_rate[text_list[self.cur_word_i]]:
                user_idxs.append(user_idx)
                u_neg_words_idxs.append(self.u_neg_word_sample[self.cur_entry_i % self.estimated_entry_size]) #in case it is larger
                product_idxs.append(product_idx)
                p_neg_words_idxs.append(self.p_neg_word_sample[self.cur_entry_i % self.estimated_entry_size])
                u_neg_p_idxs.append(self.u_neg_product_sample[self.cur_entry_i % self.estimated_entry_size])
                query_word_idxs.append(self.query_words[query_idx])
                word_idxs.append(text_list[self.cur_word_i])
                self.cur_entry_i += 1
                #padded aspect, aspect_len, value, True
                #For each word, random sample an aspect-value pair from the collected aspect-value pairs.
                #if there is none available av, a, v is none

            #move to the next
            self.cur_word_i += 1
            self.cur_epoch_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i == text_length:
                self.cur_review_i += 1
                if self.cur_review_i == self.review_size:
                    break
                self.cur_word_i = 0
                review_idx = self.train_seq[self.cur_review_i]
                user_idx = self.review_info[review_idx][0]
                product_idx = self.review_info[review_idx][1]
                query_idx = random.choice(self.product_query_idx[product_idx])
                text_list = self.review_text[review_idx]
                text_length = len(text_list)

        has_next = False if self.cur_review_i == self.review_size else True
        wrapped_neg_idxs = [u_neg_words_idxs, p_neg_words_idxs, u_neg_p_idxs]
        return user_idxs, product_idxs, query_word_idxs, word_idxs, wrapped_neg_idxs, has_next



    def get_av_train_batch(self):
        user_idxs, product_idxs, word_idxs, aspect_value_entries = [],[],[],[]
        corres_product_idxs = []
        u_neg_words_idxs, p_neg_words_idxs, u_neg_p_idxs = [],[],[]
        av_u_neg_pidxs, p_neg_aspect_idxs, pa_neg_value_idxs = [],[],[]
        query_word_idxs, corres_query_word_idxs = [], []
        review_idx = self.train_seq[self.cur_review_i]
        user_idx = self.review_info[review_idx][0]
        product_idx = self.review_info[review_idx][1]
        query_idx = random.choice(self.product_query_idx[product_idx])
        text_list = self.review_text[review_idx]
        text_length = len(text_list)
        #for each product_idx, random sample some negative products (or sample according to ranking)
        #collect aspect values from the matching of product_idx with the negative products.
        neg_product_idxs = self.neg_sample_products[self.cur_review_i]
        if self.is_feedback_same_user:
            av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=user_idx, info_level=1)
        else:
            av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=None, info_level=1)

        while len(word_idxs) < self.batch_size:
            #print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
            #if sample this word
            if self.sub_sampling_rate == None or random.random() < self.sub_sampling_rate[text_list[self.cur_word_i]]:
            #if self.sub_sampling_rate == None or self.word_sample_prob[self.cur_epoch_word_i] < self.sub_sampling_rate[text_list[self.cur_word_i]]:
                user_idxs.append(user_idx)
                u_neg_words_idxs.append(self.u_neg_word_sample[self.cur_entry_i % self.estimated_entry_size]) #in case it is larger
                product_idxs.append(product_idx)
                p_neg_words_idxs.append(self.p_neg_word_sample[self.cur_entry_i % self.estimated_entry_size])
                u_neg_p_idxs.append(self.u_neg_product_sample[self.cur_entry_i % self.estimated_entry_size])
                query_word_idxs.append(self.query_words[query_idx])
                word_idxs.append(text_list[self.cur_word_i])
                av_u_neg_pidxs.append(self.u_neg_product_av_sample[self.cur_entry_i % self.estimated_entry_size])

                if len(av_pairs) > 0:
                    aspect_value = random.choice(av_pairs)
                    aspect_value_entries.append(aspect_value)
                    #corres_product_idxs.append(product_idx)
                    #corres_query_word_idxs.append(self.query_words[query_idx])
                    p_neg_aspect_idxs.append(self.p_neg_aspect_sample[self.cur_entry_i % self.estimated_entry_size])
                    pa_neg_value_idxs.append(self.pa_neg_value_sample[self.cur_entry_i % self.estimated_entry_size])
                else:
                    aspect_value_entries.append(None)

                self.cur_entry_i += 1

                #padded aspect, aspect_len, value, True
                #For each word, random sample an aspect-value pair from the collected aspect-value pairs.
                #if there is none available av, a, v is none

            #move to the next
            self.cur_word_i += 1
            self.cur_epoch_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i == text_length:
                self.cur_review_i += 1
                if self.cur_review_i == self.review_size:
                    break
                self.cur_word_i = 0
                review_idx = self.train_seq[self.cur_review_i]
                user_idx = self.review_info[review_idx][0]
                product_idx = self.review_info[review_idx][1]
                query_idx = random.choice(self.product_query_idx[product_idx])
                text_list = self.review_text[review_idx]
                text_length = len(text_list)

                neg_product_idxs = self.neg_sample_products[self.cur_review_i]
                if self.is_feedback_same_user:
                    av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=user_idx, info_level=1)
                else:
                    av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=None, info_level=1)

        has_next = False if self.cur_review_i == self.review_size else True
        wrapped_neg_idxs = [u_neg_words_idxs, p_neg_words_idxs,
                u_neg_p_idxs, av_u_neg_pidxs, p_neg_aspect_idxs, pa_neg_value_idxs]
        return user_idxs, product_idxs, query_word_idxs, word_idxs, wrapped_neg_idxs,\
                aspect_value_entries, has_next
                #corres_query_word_idxs, corres_product_idxs, aspect_value_entries, has_next

    def prepare_test_epoch(self):
        self.test_qu_pos_product_dic = colls.defaultdict(set)
        self.test_user_query_set = set()
        #when there is relevant results in the top i for (u,q),
        #during the i+1 th iteration, (u,q) will be removed
        self.test_seq = []
        for review_idx in xrange(len(self.review_info)):
            user_idx = self.review_info[review_idx][0]
            product_idx = self.review_info[review_idx][1]
            for query_idx in self.product_query_idx[product_idx]:
                self.test_qu_pos_product_dic[(user_idx, query_idx)].add(product_idx)
                if (user_idx, query_idx) not in self.test_user_query_set:
                    self.test_user_query_set.add((user_idx, query_idx))
                    self.test_seq.append((user_idx, product_idx, query_idx, review_idx))
        #self.cur_uqr_i = 0
        self.test_qu_size = len(self.test_seq)

    def initialize_test_iter(self):
        self.cur_uqr_i = 0
        self.skip_store_rank_count = 0

    def get_test_batch(self):
        user_idxs, query_idxs, query_word_idxs = [],[],[]
        start_i = self.cur_uqr_i
        user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

        while len(user_idxs) < self.batch_size:
            user_idxs.append(user_idx)
            query_word_idxs.append(self.query_words[query_idx])
            query_idxs.append(query_idx)

            #move to the next review
            self.cur_uqr_i += 1
            if self.cur_uqr_i == len(self.test_seq):
                break
            user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

        has_next = False if self.cur_uqr_i == len(self.test_seq) else True
        return user_idxs, query_idxs, query_word_idxs, has_next

    def get_av_test_batch(self, user_ranklist_map, cur_iter_i):
        user_idxs, query_idxs, query_word_idxs = [],[],[]
        av_user_idxs, av_query_idxs, av_query_word_idxs, query_av_pairs = [],[],[],[]

        cur_av_pair_count = self.max_av_count * (cur_iter_i - 1) #will cause out of memory issue
        #cur_av_pair_count = self.max_av_count
        while len(user_idxs) + len(av_user_idxs) < self.batch_size \
                and self.cur_uqr_i < self.test_qu_size:
            av_pairs = []
            user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]
            if cur_iter_i > 1:
                uid = None
                if self.is_feedback_same_user:
                    uid = user_idx
                #print("pos_prod %s" % self.test_qu_pos_product_dic[(user_idx, query_idx)])
                #print("neg_prod %s" % user_ranklist_map[(user_idx, query_idx)][:cur_iter_i])
                av_pairs = self.item_pair_to_AVs(self.test_qu_pos_product_dic[(user_idx, query_idx)],
                        user_ranklist_map[(user_idx, query_idx)][:cur_iter_i], #neg_product_idxs
                        uid=uid, info_level=1, is_test=True)
                av_pairs = self.select_av_pairs(av_pairs, cur_av_pair_count)
            #print(len(av_pairs), av_pairs)
            if len(av_pairs) > 0:
                av_user_idxs.append(user_idx)
                av_query_idxs.append(query_idx)
                if len(av_pairs) < cur_av_pair_count:
                    query_av_pairs.append(av_pairs + [[len(self.aspect_keys) - 1, 0,
                        len(self.value_keys) - 1, -1] for _ in range(cur_av_pair_count - len(av_pairs))])
                else:
                    query_av_pairs.append(av_pairs[:cur_av_pair_count])
                #print(query_av_pairs[-1])
                #They need to align just random select some, or cutoff
                #query_av_pairs.append(random.sample(av_pairs, self.max_av_count)
                #all of them or just random select some
                av_query_word_idxs.append(self.query_words[query_idx])
            else:
                user_idxs.append(user_idx)
                query_idxs.append(query_idx)
                query_word_idxs.append(self.query_words[query_idx])
                #move to the next qu
            self.cur_uqr_i += 1
        query_av_pairs = np.asarray(query_av_pairs)
        #print(query_av_pairs.shape)

        has_next = False if self.cur_uqr_i == len(self.test_seq) else True
        return user_idxs, query_idxs, query_word_idxs,\
                av_user_idxs, av_query_idxs, av_query_word_idxs, np.asarray(query_av_pairs), has_next

    def read_train_product_ids(self, data_path):
        self.user_train_product_set_list = [set() for i in xrange(self.user_size)]
        self.train_review_size = 0
        with gzip.open("%s/train.txt.gz" % data_path , 'r') as fin:
            for line in fin:
                self.train_review_size += 1
                arr = line.strip().split('\t')
                self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))

    def get_test_product_ranklist(self, q_idx, u_idx, original_scores,
            sorted_product_idxs, cur_iter, rank_cutoff, user_ranklist_map,
            user_ranklist_score_map, keep_first_rel_qu=False):
        '''original_scores: score of each product
            sorted_product_idxs:idx ranked from lowest score to largest score
            output: user_ranklist_map, user_ranklist_score_map
        '''
        #user_ranklist_map[(u_idx, q_idx)] defaultdict(list) can be empty or a list
        rank = min(len(user_ranklist_map[(u_idx, q_idx)]), cur_iter - 1) #cur_iter from 1
        product_idx_set = set()
        #add products of previous to the filter set

        if (u_idx, q_idx) in self.test_user_query_set:
            user_ranklist_map[(u_idx, q_idx)] = user_ranklist_map[(u_idx, q_idx)][:rank]
            user_ranklist_score_map[(u_idx, q_idx)] = user_ranklist_score_map[(u_idx, q_idx)][:rank]
            #only keep the first rank
            for pid in user_ranklist_map[(u_idx, q_idx)]:
                product_idx_set.add(pid)
        else:
            #do not update in the following round
            return

        #do not store the ranklist of q,u if the first item is relevant
        is_first = True
        for idx in range(len(sorted_product_idxs)):# - 1, -1, -1): # idx from the last to 0
            product_idx = sorted_product_idxs[idx]
            if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]):
                #remove all the items the user has previously purchased in the training set
                continue
            if is_first:
                is_first = False
                if not keep_first_rel_qu and len(user_ranklist_map[(u_idx, q_idx)]) == 0 \
                        and product_idx in self.test_qu_pos_product_dic[(u_idx, q_idx)]:
                        #if the first item is relevant, do not store the results to ranklist
                        #if no items are in the dict which means this is the first iteration
                    #print self.user_ids[u_idx], q_idx, self.product_ids[product_idx]
                    self.skip_store_rank_count += 1
                    return

            if product_idx in product_idx_set:
                continue
            #product_idx_set.add(product_idx)
            #when we do padding for product set, we may have duplicate products
            user_ranklist_map[(u_idx, q_idx)].append(product_idx)
            user_ranklist_score_map[(u_idx, q_idx)].append(original_scores[product_idx])
            rank += 1
            if rank == rank_cutoff:
                break
        if user_ranklist_map[(u_idx, q_idx)][cur_iter - 1] in self.test_qu_pos_product_dic[(u_idx, q_idx)]:
            self.test_user_query_set.remove((u_idx, q_idx))
            #if the first item retrieved in this iteration is relevant, do not update in the following round
        return

    def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
        outfile = "%s/test.%s.ranklist" % (output_path, similarity_func)
        total_qu = 0
        with open(outfile, 'w') as rank_fout:
            for uq_pair in user_ranklist_map:
                if len(user_ranklist_map[uq_pair]) == 0:
                    continue
                total_qu += 1
                #print all the available uq, or uq without relevant items ranked to top iter-1.
                uidx, qidx = uq_pair
                user_id = self.user_ids[uidx]
                #map to the string id
                for i in xrange(len(user_ranklist_map[uq_pair])):
                    product_id = user_ranklist_map[uq_pair][i]
                    product_id = self.product_ids[product_id]
                    line = "%s_%d Q0 %s %d %f ProductSearchEmbedding\n" % (user_id, qidx, product_id, i+1, user_ranklist_score_map[uq_pair][i])
                    rank_fout.write(line)
        print "total qu:", total_qu

