import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from torch import LongTensor
from torch import FloatTensor
import torch.nn.functional as F
import numpy as np
import time

#there can be several types of models, each can have a class definition
class ProductSearchModel(nn.Module):
    #u->words in r of u
    #p->words in r of p
    #q, u  + p
    def prints(self, val):
        #cost a large amount of time even when debug is set to False.
        #This must have something to do with function call.
        if self.debug:
            print val
    def __init__(self, vocab_size, user_size, product_size,
            query_max_length, word_dists, product_dists,
            aspect_dists, value_dists,
            av_id2word,
            aword_idx2_aid,
            aspect_keys, value_keys,
            aspect_value_count_dic, #aspect, value:count, value:count; aspect, value:count
            window_size, embedding_size, query_weight, qnet_struct,
            comb_net_struct,
            model_net_struct, #AVHEM, HEM
            similarity_func, negative_sample,
            group_product_count = 1000,
            value_loss_func = 'softmax',
            loss_ablation = '',
            scale_grad = False,
            is_emb_sparse = False,
            is_av_embed_shared = True,
            aspect_prob_type = 'softmax',
            likelihood_way = 'av', # av|i or i|av
            use_neg_sample_per_aspect=False):
        super(ProductSearchModel, self).__init__()
        #embeddings of users, products
        #embeddings of words
        self.group_product_count = group_product_count
        self.debug = False
        self.aspect_prob_type = aspect_prob_type
        self.likelihood_way = likelihood_way
        self.vocab_size = vocab_size + 1
        self.av_vocab_size = len(av_id2word) + 1
        self.aword_idx2_aid = aword_idx2_aid
        self.loss_ablation = loss_ablation
        #add <S> end symbol to vocabulary for query
        #the word before the last word as the end symbol
        self.padding_idx = self.vocab_size - 1 # the last word for padding
        self.user_size = user_size
        self.product_size = product_size
        self.query_max_length = query_max_length
        self.qnet_struct = qnet_struct
        self.comb_net_struct = comb_net_struct
        self.prod_aspect_struct = comb_net_struct
        self.model_net_struct = model_net_struct
        self.similarity_func = similarity_func
        self.embedding_size = embedding_size
        self.is_av_embed_shared = is_av_embed_shared
        self.is_emb_sparse = is_emb_sparse
        self.use_neg_sample_per_aspect = use_neg_sample_per_aspect
        self.value_loss_func = value_loss_func

        self.softmax_func = nn.Softmax(dim = -1) # batch, 1+self.n_negs
        self.qlinear = nn.Linear(self.embedding_size, self.embedding_size) #by default bias=True
        if self.comb_net_struct == "linear_fc":
            print "train with linear_fc"
            self.word_av_fc = nn.Linear(2 * self.embedding_size, self.embedding_size) #by default bias=True
        #the concatnated semantic representation learned from words and av.

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_idx, scale_grad_by_freq = scale_grad, sparse=self.is_emb_sparse)
        self.word_bias = nn.Parameter(torch.zeros(self.vocab_size), requires_grad=True)

        self.user_emb = nn.Embedding(self.user_size, self.embedding_size, sparse=self.is_emb_sparse)
        self.product_emb = nn.Embedding(self.product_size, self.embedding_size, sparse=self.is_emb_sparse)
        self.product_bias = nn.Parameter(torch.zeros(self.product_size), requires_grad=True)
        self.n_negs = negative_sample
        self.aspect_keys = torch.LongTensor(aspect_keys)#, device = self.device)# requires_grad=False) #tensor
        self.value_keys = torch.LongTensor(value_keys)#, device = self.device)# requires_grad=False)
        self.aspect_value_words = torch.LongTensor(av_id2word)

        self.aspect_padding_idx = len(self.aspect_keys) - 1
        self.value_padding_idx = len(self.value_keys) - 1
        #tensor #self.convert2cuda_lt does not convert to gpu in the initialization
        self.aspect_bias = nn.Parameter(torch.zeros(len(self.aspect_keys)), requires_grad=True)
        self.words_dists = FloatTensor(word_dists)
        self.product_dists = FloatTensor(product_dists)
        self.aspect_dists = FloatTensor(aspect_dists)
        self.value_dists = FloatTensor(value_dists)

        if self.is_av_embed_shared:
            self.av_padding_idx = self.padding_idx
            self.av_word_emb = self.word_emb#
            self.value_bias = self.word_bias#[self.aspect_value_words]
        else:
            #set embedding for aspect and value word seperately
            # av_padding_idx can be same as the other words or not the same
            self.av_padding_idx = self.av_vocab_size - 1
            self.av_word_emb = nn.Embedding(self.av_vocab_size, self.embedding_size, padding_idx=self.av_padding_idx,scale_grad_by_freq = scale_grad)#, sparse=is_emb_sparse)
            self.value_bias = nn.Parameter(torch.zeros(self.av_vocab_size), requires_grad=True)
            #use word bias as value bias
        #separate emb to represent p(not v| a,i)
        if self.value_loss_func == 'sep_emb':
            self.neg_av_word_emb = nn.Embedding(self.av_vocab_size, self.embedding_size, padding_idx=self.av_padding_idx,scale_grad_by_freq = scale_grad)#, sparse=is_emb_sparse)
            self.neg_value_bias = nn.Parameter(torch.zeros(self.av_vocab_size), requires_grad=True)

        #aspect: (value_keys, value_distribute)
        #random sample values for each aspect may cost a lost, but not impossible
        self.value_dists_per_aspect = dict()
        for aspect in aspect_value_count_dic:
            value_keys_for_a = aspect_value_count_dic[aspect].keys()
            value_dists_for_a = FloatTensor(self.neg_distributes([aspect_value_count_dic[aspect][v] for v in value_keys_for_a]))
            self.value_dists_per_aspect[self.aword_idx2_aid[aspect]] = (np.asarray(value_keys_for_a), value_dists_for_a)

        self.query_weight = query_weight
        self.init_weights()
        self.prepare_qup_time = 0.
        self.up_loss_time = 0.
        self.uw_loss_time = 0.
        self.pw_loss_time = 0.
        self.nce_neg_gen_time = 0.
        self.nce_get_vec_time = 0.
        self.nce_compute_loss_time = 0.
        self.nce_unsqueeze_time = 0.
        self.nce_bmm_time = 0.

        self.group_ia_loss = 0.
        self.group_iav_loss = 0.
        self.group_up_loss1 = 0.
        self.group_uw_loss = 0.
        self.group_pw_loss = 0.
        self.group_up_loss2 = 0.
        self.group_av_loss = 0.
        self.group_word_loss = 0.
        self.group_iav_pos_loss = 0.
        self.group_iav_neg_loss = 0.

    def init_global_aspect_emb(self):
        #These were not converted after model.cuda()
        if self.word_emb.weight.is_cuda:
            #anything with gradients are moved to gpu after calling cuda
            #these tensors are not moved
            self.aspect_keys = self.aspect_keys.cuda()
            self.value_keys = self.value_keys.cuda()
            self.aspect_value_words = self.aspect_value_words.cuda()
            self.value_dists = self.value_dists.cuda()
            self.words_dists = self.words_dists.cuda()
            self.aspect_dists = self.aspect_dists.cuda()
            self.product_dists = self.product_dists.cuda()

    def init_aspect_vecs_for_test(self):
        self.aspect_emb = self.get_embedding_from_words(
                self.av_word_emb, self.aspect_keys, self.av_padding_idx)
        #aspect_size,  embedding_size and product_count, embedding_size
        self.aspect_size = len(self.aspect_keys)
        p_ascore_matrix = torch.mm(self.product_emb.weight, self.aspect_emb.t())
        #product_count, aspect_size
        if self.similarity_func == 'bias_product':
            p_ascore_matrix = p_ascore_matrix + self.aspect_bias
        #product_count, aspect_size
        if self.aspect_prob_type == "softmax":
            self.exp_pa_prob = self.softmax_func(p_ascore_matrix) #along dim 1
            self.exp_pa_prob = self.exp_pa_prob.log()
        elif self.aspect_prob_type == "sigmoid":
            self.exp_pa_prob = p_ascore_matrix.sigmoid().log()
        else:
            self.exp_pa_prob = p_ascore_matrix


    def neg_distributes(self, weights, distortion = 0.75):
        #print weights
        weights = np.asarray(weights)
        #print weights.sum()
        wf = weights / weights.sum()
        wf = np.power(wf, distortion)
        wf = wf / wf.sum()
        return wf

    def init_weights(self):
        self.word_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        self.user_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        self.product_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        self.word_emb.weight.data[self.padding_idx] = 0
        if self.value_loss_func == 'sep_emb':
            self.neg_av_word_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
            self.neg_av_word_emb.weight.data[self.av_padding_idx] = 0

        if not self.is_av_embed_shared:
            self.av_word_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
            self.av_word_emb.weight.data[self.av_padding_idx] = 0

        #set the row of padding_idx to 0 in the embedding matrix. (vocab_size, embed_size)
        #if we don't set the embedding of the padding index to 0, they will be randomized too.
        #the difference between the above and below ? they should be the same
    def forward(self, batch_data):
        #compute loss with av
        loss = 0
        if self.model_net_struct == 'AVHEM':
            #batch_data: user_idxs, product_idxs, query_word_idxs, word_idxs, aspect_value_entries
            user_idxs, product_idxs, query_word_idxs, word_idxs,\
                    wrapped_neg_idxs, aspect_value_entries = batch_data
            av_loss = self.qup_av_nce_loss(batch_data)
            batch_data = [user_idxs, product_idxs, query_word_idxs, word_idxs] + [wrapped_neg_idxs[:3]]
            word_loss = self.qup_nce_loss(batch_data)
            #compute loss with av
            loss = av_loss + word_loss
            self.group_av_loss += av_loss.item()
            self.group_word_loss += word_loss.item()
        else: #HEM
            loss = self.qup_nce_loss(batch_data)
        return loss

    def get_product_scores(self, user_idxs, query_word_idxs, product_idxs = None):
        # product_idxs can be specified or all the availale products in the category
        # get user embedding [None, embed_size]
        query_word_idxs, user_idxs = map(self.convert2cuda_lt,
                [query_word_idxs, user_idxs],
                [True, True]) #it is for evaluation
        query_vecs = self.get_embedding_from_words(self.word_emb, query_word_idxs, self.padding_idx)
        # get query embedding [batch_size, embed_size]
        batch_size = len(user_idxs)
        user_vecs = self.user_emb(user_idxs)
        product_vecs = None
        product_bias = None
        if product_idxs is not None:
            #batch_size, max_purchased_product_count
            product_idxs = self.convert2cuda_lt(product_idxs, evaluation=True)
            product_vecs = self.product_emb(product_idxs)
            #batch, max_purchased, embed_size
            product_bias = self.product_bias[product_idxs.view(-1)].view(batch_size, -1)
        else:
            product_vecs = self.product_emb.weight #weight is a variable, weight.data is a tensor
            product_bias = self.product_bias
        # get candidate product embedding [None, embed_size]
        qu_vectors = user_vecs
        if self.query_weight > 0:
            qu_vectors = self.query_weight * query_vecs + (1.0 - self.query_weight) * user_vecs

        qu_pscores = self.compute_sim(qu_vectors, product_vecs, product_bias, self.similarity_func)
        return qu_pscores.data.cpu().numpy()

    def pav_scores_sep_emb(self, product_vecs, aspect_vecs, value_idxs, value_symbols, aspect_masks):
        value_vecs = self.av_word_emb(value_idxs).permute(0,2,1) #batch_size, embed_size, aspect_size
        batch_size, embed_size, max_av_count = value_vecs.size()
        neg_value_vecs = self.neg_av_word_emb(value_idxs).permute(0,2,1) #batch_size, embed_size, aspect_size
        if self.word_emb.weight.is_cuda:
            pa_value_scores = torch.zeros(batch_size, self.product_size).cuda()
        else:
            pa_value_scores = torch.zeros(batch_size, self.product_size)
        s_pidx = 0
        while s_pidx < self.product_size:
            e_pidx = min(self.product_size, s_pidx + self.group_product_count)
            product_vecs_slice = product_vecs[s_pidx:e_pidx, :] #get vecs
            product_slice_size = product_vecs_slice.size()[0]
            expand_prod_vecs = product_vecs_slice.unsqueeze(0).unsqueeze(-2).expand(batch_size, -1,  max_av_count, -1)
            expand_aspect_vecs = aspect_vecs.unsqueeze(1).expand(-1, product_slice_size, -1, -1)
            #batch_size, product_count, aspect_size,embedding_size
            if self.prod_aspect_struct == 'add':
                product_aspect_vecs = expand_prod_vecs + expand_aspect_vecs
            elif self.prod_aspect_struct == 'mul':
                product_aspect_vecs = expand_prod_vecs * expand_aspect_vecs
            else:
                product_aspect_vecs = torch.tanh(self.word_av_fc(
                    torch.cat([expand_aspect_vecs, expand_prod_vecs], -1)))
            #batch_size, product_slice_size, aspect_size, value_size
            product_aspect_vecs = product_aspect_vecs.view(batch_size, -1, embed_size)

            pa_value_prob = torch.bmm(product_aspect_vecs, value_vecs).view(batch_size, -1, max_av_count, max_av_count)
            neg_pa_value_prob = torch.bmm(product_aspect_vecs, neg_value_vecs).view(batch_size, -1, max_av_count, max_av_count)
            if self.aspect_prob_type == "sigmoid":
                pa_value_prob = pa_value_prob.sigmoid().log()
                neg_pa_value_prob = neg_pa_value_prob.sigmoid().log()
            #batch_size * product_slice_size

            for i in xrange(batch_size):
                for j in xrange(max_av_count):
                    if value_symbols[i][j] == 1:
                        pa_value_scores[i,s_pidx:e_pidx] += pa_value_prob[i, :, j, j] * aspect_masks[i,j]
                    elif value_symbols[i][j] == -1:
                        if self.loss_ablation == 'negv':
                            pass
                        elif self.loss_ablation == 'sepv':
                            pa_value_scores[i,s_pidx:e_pidx] += (1 - pa_value_prob[i, :, j, j]) * aspect_masks[i,j]
                        else:
                            pa_value_scores[i,s_pidx:e_pidx] += neg_pa_value_prob[i, :, j, j] * aspect_masks[i,j]

            s_pidx += self.group_product_count

        return pa_value_scores

    def pav_scores_softmax(self, product_vecs, aspect_vecs, value_idxs, value_symbols, aspect_masks):
        batch_size, max_av_count = value_idxs.size()
        if self.word_emb.weight.is_cuda:
            pa_value_scores = torch.zeros(batch_size, self.product_size).cuda()
        else:
            pa_value_scores = torch.zeros(batch_size, self.product_size)
        s_pidx = 0
        value_vecs = self.av_word_emb(self.value_keys)
        value_symbol_bias = (value_symbols - 1) * (-0.5) #1->0, -1->1
        #value_symbols: batch_size, max_av_count 1 -> x, -1 -> 1 - x
        while s_pidx < self.product_size:
            e_pidx = min(self.product_size, s_pidx + self.group_product_count)
            product_vecs_slice = product_vecs[s_pidx:e_pidx, :] #get vecs
            product_slice_size = product_vecs_slice.size()[0]
            expand_prod_vecs = product_vecs_slice.unsqueeze(0).unsqueeze(-2).expand(batch_size, -1,  max_av_count, -1)
            expand_aspect_vecs = aspect_vecs.unsqueeze(1).expand(-1, product_slice_size, -1, -1)
            #batch_size, product_count, aspect_size,embedding_size
            if self.prod_aspect_struct == 'add':
                product_aspect_vecs = expand_prod_vecs + expand_aspect_vecs
            elif self.prod_aspect_struct == 'mul':
                product_aspect_vecs = expand_prod_vecs * expand_aspect_vecs
            else:
                product_aspect_vecs = torch.tanh(self.word_av_fc(
                    torch.cat([expand_aspect_vecs, expand_prod_vecs], -1)))
            #batch_size, product_slice_size, aspect_size, value_size
            pa_value_prob = self.softmax_func(torch.matmul(product_aspect_vecs, value_vecs.t()))
            #batch_size * product_slice_size
            #print(pa_value_prob.size())

            for i in xrange(batch_size):
                for j in xrange(max_av_count):
                    if not aspect_masks[i,j]:
                        continue
                    pa_value_scores[i,s_pidx:e_pidx] += \
                            (pa_value_prob[i, :, j, value_idxs[i, j]] * value_symbols[i,j] \
                            + value_symbol_bias[i,j]).log()

            s_pidx += self.group_product_count
        return pa_value_scores

    def get_product_scores_av(self, user_idxs, query_word_idxs, \
            aspect_value_entries, product_idxs = None):
        # get user embedding [None, embed_size]
        #query_av_pairs: batch_size, max_av_count, 4 ([aspect_idx, aspect_len, value_idxs, value_symbols])
        aspect_idxs = aspect_value_entries[:,:,0]
        value_idxs = aspect_value_entries[:,:,2]
        value_symbols = aspect_value_entries[:,:,3]
        query_word_idxs, user_idxs, aspect_idxs, value_idxs = map(
                self.convert2cuda_lt,
                [query_word_idxs, user_idxs, aspect_idxs, value_idxs])
        query_vecs = self.get_embedding_from_words(self.word_emb, query_word_idxs, self.padding_idx)
        user_vecs = self.user_emb(user_idxs)
        batch_size, max_av_count = aspect_idxs.size()
        aspect_idxs = aspect_idxs.view(-1) #batch_size * max_av_count
        product_vecs = None
        product_bias = None
        if product_idxs is not None:
            #batch_size, max_purchased_product_count
            product_idxs = self.convert2cuda_lt(product_idxs, evaluation=True)
            product_vecs = self.product_emb(product_idxs)
            product_bias = self.product_bias[product_idxs.view(-1)].view(batch_size, -1)
        else:
            product_vecs = self.product_emb.weight #weight is a variable, weight.data is a tensor
            product_bias = self.product_bias
        # get candidate product embedding [None, embed_size]
        #print "product_vecs.size", product_vecs.size()
        qu_vectors = user_vecs
        if self.query_weight > 0:
            qu_vectors = self.query_weight * query_vecs + (1.0 - self.query_weight) * user_vecs

        #u+q->i, i->a, i+a->v
        #logp(i|u+q)+logp(a|i)+logp(v|i,a)
        qu_pscores = self.compute_sim(qu_vectors, product_vecs, product_bias, self.similarity_func)

        aspect_vecs = self.aspect_emb[aspect_idxs].view(batch_size, max_av_count, -1)#
        #batch_size * max_av_count, embedding_size
        #padding aspects will get zero vectors.
        p_ascore_matrix = self.exp_pa_prob[:,aspect_idxs] # product_size, batch_size * max_av_count
        #batch_size, product_size, embedding_size
        p_ascore_matrix = p_ascore_matrix.view(-1, batch_size, max_av_count).permute(1,0,2)
        #batch_size, product_count, max_av_count
        aspect_masks = (aspect_idxs.view(batch_size, max_av_count) != self.aspect_padding_idx).float()

        p_ascores = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        if not self.loss_ablation == "ia":
            p_ascores = torch.sum(p_ascore_matrix * aspect_masks.unsqueeze(1), -1) #/ aspect_masks.unsqueeze(1).sum(-1)
        #divided by aspect length doesn't make it better.
        #cost too much memory
        pa_value_scores = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        if not self.loss_ablation == 'iav':
            if (not self.value_loss_func == 'softmax') or self.loss_ablation == 'negv' or self.loss_ablation == 'sepv':
                pa_value_scores = self.pav_scores_sep_emb(product_vecs, aspect_vecs, value_idxs, value_symbols, aspect_masks)
            else:#don't use this ever
                pa_value_scores = self.pav_scores_softmax(product_vecs, aspect_vecs, value_idxs, value_symbols, aspect_masks)

        scores = qu_pscores + p_ascores  + pa_value_scores
        return scores.data.cpu().numpy()

    def compute_sim(self, qu_vectors, product_vecs, product_bias, sim_func):
        qu_pscores = None
        if len(product_vecs.size()) == 2: #all the products
            if sim_func == 'product':
                qu_pscores = torch.mm(qu_vectors, product_vecs.t())#batch_size, product_count
            elif sim_func == 'bias_product':
                qu_pscores = torch.mm(qu_vectors, product_vecs.t()) + product_bias
                #batch_size, 2e ; 2e, product_size; product_size
            else:
                qu_vectors = qu_vectors / torch.norm(qu_vectors, p=2, dim=1, keepdim=True)
                #or qu_vectors.div(torch.norm(....))
                #l2 norm along column, (batch_size,1)
                product_vecs = product_vecs / torch.norm(product_vecs, p=2, dim=1, keepdim=True)
                qu_pscores = torch.mm(qu_vectors, product_vecs.t())#batch_size, product_count
        else: # 3 dimensions, batch_size, max_product_size, embedding_size
            if sim_func == 'product':
                qu_pscores = torch.bmm(product_vecs, qu_vectors.unsqueeze(2)).squeeze() #batch_size, product_count
            elif sim_func == 'bias_product':
                qu_pscores = torch.bmm(product_vecs, qu_vectors.unsqueeze(2)).squeeze() + product_bias
            else:
                qu_vectors = qu_vectors / torch.norm(qu_vectors, p=2, dim=1, keepdim=True)
                #batch_size, embedding_size
                #or qu_vectors.div(torch.norm(....))
                #l2 norm along column, (batch_size,1)
                product_vecs = product_vecs / torch.norm(product_vecs, p=2, dim=2, keepdim=True)
                #batch_size, product_size, embedding_size
                qu_pscores = torch.bmm(product_vecs, qu_vectors.unsqueeze(2)).squeeze() #batch_size, product_count
        return qu_pscores

    def convert2cuda_ft(self, vals):#convert array to cuda float tensor
        vals = FloatTensor(vals)#, requires_grad=False)
        vals = vals.cuda() if self.word_emb.weight.is_cuda else vals
        return vals

    def convert2cuda_lt(self, idxs, evaluation=False):#convert index array to cuda
        if type(idxs) is not torch.Tensor:
            idxs = LongTensor(idxs)
        if self.word_emb.weight.is_cuda and not idxs.is_cuda:
        #by default, requires_grad of nn.Parameter is True
            idxs = idxs.cuda()
        return idxs

    def get_fs_from_words(self, word_emb, word_idxs, padding_idx):
        #word_idxs need to be tensor instead of ndarray
        # get mean word vectors
        #batch, max_query_length
        word_vecs = word_emb(word_idxs)
        #batch, max_query_length, emb_size
        masks = (word_idxs != padding_idx).float()
        masks = masks.unsqueeze(2)
        #batch, max_query_length,1
        seq_lens =  masks.sum(1)
        seq_lens[(seq_lens==0).nonzero()] = 1 #in case all the word idxs are padding idx
        word_vecs = torch.sum(word_vecs * masks, 1) / seq_lens #in case division by 0
        #batch, emb_size
        f_s = torch.tanh(self.qlinear(word_vecs))
        return f_s

    def get_summation_from_words(self, word_emb, word_idxs, padding_idx):
        # get mean word vectors
        #batch, max_query_length
        word_vecs = word_emb(word_idxs)
        #batch, max_query_length, emb_size
        masks = (word_idxs != padding_idx).float()
        masks = mask.unsqueeze(2)
        #batch, max_query_length,1
        seq_lens =  masks.sum(1)
        seq_lens[(seq_lens==0).nonzero()] = 1
        word_vecs = torch.sum(word_vecs * masks, 1) / seq_lens
        #batch, emb_size
        return word_vecs

    def get_attention_from_words(self, word_idxs):
        pass

    def get_embedding_from_words(self, word_emb, word_idxs, padding_idx):
        if 'mean' in self.qnet_struct: # mean vector
            #print('Query model: mean')
            return self.get_summation_from_words(word_emb, word_idxs, padding_idx)
        elif 'fs' in self.qnet_struct: # LSE f(s)
            #print('Query model: LSE f(s)')
            return self.get_fs_from_words(word_emb, word_idxs, padding_idx)
        else:
            #print('Query model: Attention')
            return self.get_attention_from_words(word_idxs)

    def sample_neg_values(self, batch_size, aspect_idxs):
        if self.use_neg_sample_per_aspect:
            neg_values = []
            for i in range(batch_size):
                value_keys_for_a, value_dists_for_a = self.value_dists_per_aspect[aspect_idxs[i]]
                if value_symbols[i] < 0 and len(value_keys_for_a) == 1:
                    #this is a negative value and only one value for this aspect.
                    sample_idxs = torch.multinomial(self.value_dists, self.n_negs, replacement=True)
                else:
                    sample_idxs = torch.multinomial(value_dists_for_a, self.n_negs, replacement=True)
                neg_values.extend(value_keys_for_a[sample_idxs])
            neg_values = self.convert2cuda_lt(neg_values)
                #word ids in the av vocab if not shared or word vocab if shared
        else:
            if self.value_dists is not None:
                neg_sample_idxs = torch.multinomial(self.value_dists, batch_size * self.n_negs, replacement=True)
            else:
                if self.word_emb.weight.is_cuda:
                    neg_sample_idxs = FloatTensor(batch_size  * self.n_negs).cuda().uniform_(0, len(self.value_keys) - 1).long()
                else:
                    neg_sample_idxs = FloatTensor(batch_size  * self.n_negs).uniform_(0, len(self.value_keys) - 1).long()
                #neg_sample_idxs = self.convert2cuda_lt(neg_sample_idxs)
            neg_values = self.value_keys[neg_sample_idxs] #word id in the av vocabulary
        return neg_values


    '''
        i + a -> v
        source_vectors from i and a.
        given av weight distribution
        loss for pos v: -log P(v|i,a)
        loss for neg v: -log P(not v| i, a)
        sample negative samples for each positive sample according to weight distribution
        The other way is to do it in a opposite way as negative sampling
        max(log(sigmoid(-(ai * v-))) + sum_{v+ \in V+}(log(sigmoid(ai* v+))))
        max(log(sigmoid((ai * v+))) + sum_{v- \in V-}(log(sigmoid(-ai* v-))))
    '''
    def neg_log_likelihood_loss_sep_emb(self, aspect_idxs, source_vectors,
            target_idxs, pos_idxs, neg_idxs, target_emb, target_bias,
            neg_target_emb, neg_target_bias, neg_sample_idxs = None, sim_func='product'):

        batch_size = source_vectors.size()[0]
        if neg_sample_idxs is not None:
            neg_values = self.value_keys[neg_sample_idxs] #word id in the av vocabulary
        else:
            neg_values = self.sample_neg_values(batch_size, aspect_idxs)

        pos_value_loss = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        neg_value_loss = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        if len(pos_idxs) > 0:
            pos_value_loss = self.part_loss(pos_idxs, target_idxs, neg_values,
                    source_vectors, target_emb, target_bias, sim_func=sim_func)
        if len(neg_idxs) > 0 and not self.loss_ablation == 'negv': #neg value not ablated
            neg_value_loss = self.part_loss(neg_idxs, target_idxs, neg_values,
                    source_vectors, neg_target_emb, neg_target_bias, sim_func=sim_func)
        self.group_iav_pos_loss += pos_value_loss.item()
        self.group_iav_neg_loss += neg_value_loss.item()

        loss = pos_value_loss + neg_value_loss
        #batch_size, 1
        return loss

    def part_loss(self, idxs, target_idxs, neg_values, source_vectors, target_emb, target_bias, sim_func='product'):
        batch_size = idxs.size()[0]
        source_vectors = source_vectors[idxs].unsqueeze(2)
        target_vectors = target_emb(self.value_keys[target_idxs[idxs]]).unsqueeze(1)
        true_bias = target_bias[self.value_keys[target_idxs[idxs]]].view(batch_size, -1) #batch_size, 1
        neg_bias = target_bias[neg_values].view(-1, self.n_negs) #batch_size, self.n_negs
        neg_bias = neg_bias[idxs]

        cur_neg_values = neg_values.view(-1, self.n_negs)
        nvectors = target_emb(cur_neg_values[idxs]).unsqueeze(1)
        #64, 100, 1
        #batch_size, 1, embedding_size # summation of those words
        #64, 1, 100
        #batch, n_negs, embed_size
        oloss = torch.bmm(target_vectors, source_vectors).squeeze(-1)
        #batch_size,1
        #batch_size, n_negs
        #sigmoid functions
        nloss = torch.bmm(nvectors.neg().view(batch_size, self.n_negs, -1), source_vectors).squeeze(-1)
        #if self.similarity_func == 'bias_product':
        if sim_func == 'bias_product':
            oloss = oloss + true_bias
            nloss = nloss + neg_bias
        #batch_size, n_negs, 1
        oloss = oloss.sigmoid().log()
        nloss = nloss.sigmoid().log().sum(1)
        loss = -(oloss + nloss).mean() #keep batch size
        return loss

    def neg_log_likelihood_loss(self, aspect_idxs, source_vectors,
            target_idxs, value_symbols, target_emb, target_bias,
            value_loss_func='softmax', neg_sample_idxs = None):
        batch_size = source_vectors.size()[0]
        #print batch_size

        if neg_sample_idxs is not None:
            neg_values = self.value_keys[neg_sample_idxs] #word id in the av vocabulary
        else:
            neg_values = self.sample_neg_values(batch_size, aspect_idxs)

        #batch_size, embedding_size, 1 # the last one was from unsqueezed
        #print "source:", source_vectors.size()
        #64, 100, 1
        source_vectors = source_vectors.unsqueeze(2)
        target_vectors = target_emb(self.value_keys[target_idxs]).unsqueeze(1)
        true_bias = target_bias[self.value_keys[target_idxs]].view(batch_size, -1) #batch_size, 1
        neg_bias = target_bias[neg_values].view(batch_size, -1) #batch_size, self.n_negs
        #batch_size, 1, embedding_size # summation of those words
        #64, 1, 100
        neg_values = neg_values.view(batch_size, -1)
        nvectors = target_emb(neg_values)

        #batch, n_negs, embed_size
        oloss = torch.bmm(target_vectors, source_vectors).squeeze(-1)
        #batch_size,1
        #in case that batch_size == 1, specify the dimension to squeeze, do not just use squeeze
        #batch_size, n_negs
        if value_loss_func == 'softmax' or not self.loss_ablation == 'sepv':
            nloss = torch.bmm(nvectors.view(batch_size, self.n_negs, -1), source_vectors).squeeze(-1)
            if self.similarity_func == 'bias_product':
                oloss = oloss + true_bias
                nloss = nloss + neg_bias
            dot_all = torch.cat([oloss, nloss], 1) #batch_size, 1+self.n_negs
            output_prob = self.softmax_func(dot_all)
            output_prob = output_prob[:,0] * value_symbols #get the first one of each entry in the batch
            #1->x,-1->-x
            value_symbols = (value_symbols - 1) * (-0.5)
            #1->0, -1->1
            output_prob = output_prob + value_symbols
            #1->x, -1->1-x
            loss = output_prob.log().neg().mean()
        else:
            nloss = torch.bmm(nvectors.neg().view(batch_size, self.n_negs, -1), source_vectors).squeeze(-1)
            #batch_size, n_negs
            oloss = oloss * value_symbols
            nloss = nloss * value_symbols.unsqueeze(-1)
            if self.similarity_func == 'bias_product':
                oloss = oloss + true_bias
                nloss = nloss + neg_bias
            #batch_size, n_negs, 1
            oloss = oloss.sigmoid().log()
            nloss = nloss.sigmoid().log().sum(1)
            loss = -(oloss + nloss).mean() #keep batch size
        #batch_size, 1
        #64, 1,1 -> 64
        #when use context, give a dimension as a parameter to mean
        return loss

    def qup_av_nce_loss(self, batch_data):
        '''
        q+u->i; i->a; i+a->v or -v
        '''
        #aspect_idxs must be nonempty
        user_idxs, product_idxs, query_word_idxs, _,\
                wrapped_neg_idxs, aspect_value_entries = batch_data
        av_u_neg_pidxs, p_neg_aspect_idxs, pa_neg_value_idxs \
                = map(self.convert2cuda_lt, wrapped_neg_idxs[3:])
        #[aspect, aspect_len, self.vword_idx2_vid[val[0]], False]
        aspect_idxs = [x[0] for x in aspect_value_entries if x] #multiple idx for one aspect
        value_idxs = [x[2] for x in aspect_value_entries if x] # value_id: single idx for one value
        value_symbols = [x[3] for x in aspect_value_entries if x] # pos or neg value
        corres_product_idxs = [product_idxs[i] for i in xrange(len(product_idxs)) if aspect_value_entries[i]]
        valid_idxs = [i for i in xrange(len(product_idxs)) if aspect_value_entries[i]]

        user_idxs, product_idxs, corres_product_idxs, query_word_idxs, valid_idxs \
                = map(self.convert2cuda_lt, \
                [user_idxs, product_idxs, corres_product_idxs, query_word_idxs, valid_idxs])
        #get query embeddings
        query_vecs = self.get_embedding_from_words(self.word_emb, query_word_idxs, self.padding_idx)
        user_vecs = self.user_emb(user_idxs)
        product_vecs = self.product_emb(corres_product_idxs)
        ia_loss = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        iav_loss = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        up_loss = torch.tensor(0.).cuda() if self.word_emb.weight.is_cuda else torch.tensor(0.)
        if len(aspect_idxs) > 0:
            aspect_idxs, value_idxs = map(self.convert2cuda_lt, [aspect_idxs, value_idxs])
            value_symbols = self.convert2cuda_ft(value_symbols)

            pos_idxs = (value_symbols == 1).nonzero()[:,0]
            neg_idxs = (value_symbols == -1).nonzero()[:,0]
            aspect_vecs = self.get_embedding_from_words(
                    self.av_word_emb, self.aspect_keys[aspect_idxs], self.av_padding_idx)

            if "av" in self.likelihood_way:
                if not self.loss_ablation == "ia":
                    #if pa is not ablated
                    ia_loss = self.nce_loss(product_vecs, self.aspect_dists,
                            aspect_idxs, self.av_word_emb, self.aspect_bias,
                            idx2target = self.aspect_keys, neg_sample_idxs = p_neg_aspect_idxs)

                if self.prod_aspect_struct == 'add':
                    p_aspect_vecs = aspect_vecs + product_vecs #batch_size, embed_size
                elif self.prod_aspect_struct == 'mul':
                    p_aspect_vecs = aspect_vecs * product_vecs #batch_size, embed_size
                else:
                    p_aspect_vecs = torch.tanh(self.word_av_fc(torch.cat([aspect_vecs, product_vecs], 1)))

                if not self.loss_ablation == "iav":
                    if (self.value_loss_func == 'sep_emb' \
                            and not self.loss_ablation == 'sepv') or self.loss_ablation == 'negv':
                        iav_loss = self.neg_log_likelihood_loss_sep_emb(aspect_idxs, p_aspect_vecs,
                                value_idxs, pos_idxs, neg_idxs, self.av_word_emb, self.value_bias,
                                self.neg_av_word_emb, self.neg_value_bias, neg_sample_idxs = pa_neg_value_idxs)
                    else: #self.loss_ablation == sepv when value_loss_func should be sigmoid
                        iav_loss = self.neg_log_likelihood_loss(aspect_idxs, p_aspect_vecs,
                                value_idxs, value_symbols, self.av_word_emb, self.value_bias,
                                self.value_loss_func, neg_sample_idxs = pa_neg_value_idxs)

        query_vecs = self.query_weight * query_vecs + (1.0 - self.query_weight) * user_vecs
        if self.value_loss_func == 'sep_emb' and "item" in self.likelihood_way and len(aspect_idxs) > 0: #or av likelihood
            query_vecs[valid_idxs] = query_vecs[valid_idxs] + 0.5 * aspect_vecs
            if len(pos_idxs) > 0:
                query_vecs[valid_idxs][pos_idxs] = query_vecs[valid_idxs][pos_idxs] + 0.5 * self.av_word_emb(self.value_keys[value_idxs[pos_idxs]])
            if len(neg_idxs) > 0:
                query_vecs[valid_idxs][neg_idxs] = query_vecs[valid_idxs][neg_idxs] + 0.5 * self.neg_av_word_emb(self.value_keys[value_idxs[neg_idxs]])

        if "ori" in self.likelihood_way or "item" in self.likelihood_way:
            up_loss = self.nce_loss(query_vecs, self.product_dists,
                    product_idxs, self.product_emb, self.product_bias,
                    neg_sample_idxs = av_u_neg_pidxs, sim_func = self.similarity_func)

        self.group_ia_loss += ia_loss.item()
        self.group_iav_loss += iav_loss.item()
        self.group_up_loss1 += up_loss.item()

        loss = ia_loss + iav_loss + up_loss
        #the sum of the three parts averaged over a batch
        #l2 loss will be added by the optimizer
        return loss

    def qup_nce_loss(self, batch_data):
        #get query embedding
        start_time = time.time()
        user_idxs, product_idxs, query_word_idxs, word_idxs \
                = map(self.convert2cuda_lt, batch_data[:-1])
        u_neg_words_idxs, p_neg_words_idxs, u_neg_p_idxs = map(self.convert2cuda_lt, batch_data[-1])

        query_vecs = self.get_embedding_from_words(self.word_emb, query_word_idxs, self.padding_idx)
        user_vecs = self.user_emb(user_idxs)
        product_vecs = self.product_emb(product_idxs)
        self.prepare_qup_time += time.time() - start_time
        start_time = time.time()
        #different types of representations
        #get product embeddings
        #get user embeddings
        uw_loss = self.nce_loss(user_vecs, self.words_dists,
                word_idxs, self.word_emb, self.word_bias, neg_sample_idxs=u_neg_words_idxs)
        self.uw_loss_time += time.time() - start_time
        start_time = time.time()
        pw_loss = self.nce_loss(product_vecs, self.words_dists,
                word_idxs, self.word_emb, self.word_bias, neg_sample_idxs=p_neg_words_idxs)

        start_time = time.time()
        if self.query_weight > 0:
            query_vecs = self.query_weight * query_vecs + (1.0 - self.query_weight) * user_vecs
        self.pw_loss_time += time.time() - start_time

        up_loss = self.nce_loss(query_vecs, self.product_dists,
                product_idxs, self.product_emb, self.product_bias, neg_sample_idxs=u_neg_p_idxs)

        self.group_uw_loss += uw_loss.item()
        self.group_pw_loss += pw_loss.item()
        self.group_up_loss2 += up_loss.item()

        loss = uw_loss + pw_loss + up_loss
        self.up_loss_time += time.time() - start_time
        #the sum of the three parts averaged over a batch

        #calculate the uqp loss
        #calculate the u->word in review loss
        #calculate the p->word in review loss
        #l2 loss will be added by the optimizer
        return loss


    def nce_loss(self, source_vectors, weight_dists, target_idxs,
            target_emb, target_bias, neg_sample_idxs = None,
            idx2target = None, sim_func = 'product'):
        #given inner embeddings and outer embeddings
        #uids, pids, word_idx
        #neg_count
        #idx2target: idx to actual target idxs. 1 (aspect id) -> 2,4 (word id)
        #context_size = 1 for u-p, u-w, p-w neg sampling
        start_time = time.time()
        context_size = 1
        target_size = target_bias.size()[0]#target_size , 1
        '''
            given word weight distribution
            or product weight distribution
            sample negative samples for each positive sample according to weight distribution
            calculate the loss for skip-gram negative-sampling
            the function can be vec(u) * vec(p)->score or [u,p] -> nn -> score
            the function can be vec(u and q) * vec(p)->score or [u,p] -> nn -> score
        '''
        batch_size = source_vectors.size()[0]

        #print batch_size
        if neg_sample_idxs is None:
            if weight_dists is not None:
                neg_sample_idxs = torch.multinomial(weight_dists, batch_size * context_size * self.n_negs, replacement=True)
                #cuda.longtensor
                #negative sampling according to x^0.75
                #each context word has n_neg corresponding samples
            else:
                if source_vectors.is_cuda:
                    neg_sample_idxs = FloatTensor(batch_size * context_size * self.n_negs, device='gpu:0').uniform_(0, target_size - 1).long()
                else:
                    neg_sample_idxs = FloatTensor(batch_size * context_size * self.n_negs, device='cpu').uniform_(0, target_size - 1).long()
            #convert
        else:
            neg_sample_idxs = neg_sample_idxs.view(-1)
        self.nce_neg_gen_time += time.time() - start_time
        start_time = time.time()
        neg_bias = target_bias[neg_sample_idxs].view(batch_size, -1)
        true_bias = target_bias[target_idxs].view(batch_size, -1)
        #batch_size, context_size * n_negs, embedding_size
        #64,5,100
        if idx2target is not None:
            target_vectors = self.get_embedding_from_words(
                    target_emb, idx2target[target_idxs], self.av_padding_idx)
            target_vectors = target_vectors.view(batch_size, 1, -1)
            neg_vectors = self.get_embedding_from_words(
                    target_emb, idx2target[neg_sample_idxs], self.av_padding_idx)
            nvectors = neg_vectors.view(batch_size, context_size * self.n_negs, -1)
        else:
            target_vectors = target_emb(target_idxs).unsqueeze(1)
            neg_sample_idxs = neg_sample_idxs.view(batch_size, -1)
            nvectors = target_emb(neg_sample_idxs)
        #batch_size, context_size * n_negs, embedding_size
        self.nce_get_vec_time += time.time() - start_time
        start_time = time.time()
        #batch_size, embedding_size
        #when an object of this class call cuda(), all the tensors will be moved to gpu,
        #those appearing only in the parameters should be converted to cuda tensors.

        self.nce_unsqueeze_time += time.time() - start_time
        start_time = time.time()

        oloss = torch.bmm(target_vectors, source_vectors.unsqueeze(2)).squeeze(-1)
        nloss = torch.bmm(nvectors.neg(), source_vectors.unsqueeze(2)).squeeze(-1)
        #batch_size, context_size * n_negs #, 1
        #sum along dim 2 (n_negs), mean along dim 1, which is context size
        #batch_size, 1
        self.nce_bmm_time += time.time() - start_time
        start_time = time.time()
        if sim_func == 'bias_product':
            oloss = oloss + true_bias
            nloss = nloss + neg_bias
        #cosine similarity or product or bias_product
        oloss = oloss.sigmoid().log()#.mean()
        nloss = nloss.sigmoid().log().sum(1)# context_size, .mean(1)
        #64, 1,1 -> 64
        #when use context, give a dimension as a parameter to mean
        #log(sigmod(batch_size, context_size))
        #the mean value along dim 1 -> batch, 1
        loss = -(oloss + nloss).mean()
        self.nce_compute_loss_time += time.time() - start_time
        return loss

def test():
    #model = ProductSearchModel()
    pass
if __name__ == '__main__':
    test()


