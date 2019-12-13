from __future__ import print_function
import shutil
import torch
import math
import os
import random
import sys
import time
import argparse
import numpy as np
import collections as colls

from decimal import Decimal
from six.moves import xrange    # pylint: disable=redefined-builtin
from model import ProductSearchModel
import data_util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_learning_rate", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.90,
                            help="Learning rate decays by this much.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                            help="Clip gradients to this norm.")
    parser.add_argument("--subsampling_rate", type=float, default=1e-5,
                            help="The rate to subsampling.")
    parser.add_argument("--L2_lambda", type=float, default=0.0,
                            help="The lambda for L2 regularization.")
    parser.add_argument("--query_weight", type=float, default=0.5,
                            help="The weight for query.")
    parser.add_argument("--batch_size", type=int, default=64,
                            help="Batch size to use during training.")
#rank list size should be read from data
    parser.add_argument("--data_dir", type=str, default="/tmp", help="Data directory")
    parser.add_argument("--input_train_dir", type=str, default="", help="The directory of training and testing data")
    parser.add_argument("--save_dir", type=str, default="/tmp", help="Model directory & output directory")
    parser.add_argument("--similarity_func", type=str, default="product", help="Select similarity function, which could be product, cosine and bias_product")
    parser.add_argument("--qnet_struct", type=str, default="simplified_fs", help="Specify network structure parameters. Please read readme.txt for details.")
    parser.add_argument("--comb_net_struct", type=str, default="add", help="Specify network structure parameters. Please read readme.txt for details.")
    parser.add_argument("--model_net_struct", type=str, default="AVHEM", help="Specify network structure parameters. HEM, AVHEM")
    parser.add_argument("--value_loss_func", type=str, default="softmax", help="softmax or neg sigmoid for negative value estimation")
    parser.add_argument("--aspect_prob_type", type=str, default="softmax", help="softmax or neg sigmoid for aspect estimation")
    parser.add_argument("--likelihood_way", type=str, default="av_ori", help="av or item or av_item for training", choices=["av", "item", "av_item", "av_ori"])
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd or adam", choices = ["sgd", "adam"])
    parser.add_argument("--embed_size", type=int, default=100, help="Size of each embedding.")
    parser.add_argument("--window_size", type=int, default=5, help="Size of context window.")
    parser.add_argument("--max_train_epoch", type=int, default=20,
                            help="Limit on the epochs of training (0: no limit).")
    parser.add_argument("--steps_per_checkpoint", type=int, default=400,
                            help="How many training steps to do per checkpoint.")
    parser.add_argument("--seconds_per_checkpoint", type=int, default=3600,
                            help="How many seconds to wait before storing embeddings.")
    parser.add_argument("--negative_sample", type=int, default=5,
                            help="How many samples to generate for negative sampling.")
    parser.add_argument("--loss_ablation", type=str, default='', choices=["","none", "ia", "iav", "negv", "sepv"],
                            help="remove nothing, item->aspect, item+aspect->value, neg values, separate embeddings.")
    parser.add_argument("--is_feedback_same_user", action='store_true',
                            help="feedback from the same user.")
    parser.add_argument("--keep_first_rel_qu", action='store_true',
                            help="if the first qrel is relevant, keep this qu or not.")
    parser.add_argument("--neg_per_pos", type=int, default=2,
                            help="How many negative samples used to pair with postive results.")
    parser.add_argument("--threshold", type=str, default='strict', choices=["strict","relax"],
                            help="Use lexicon generated from strict or relax condition.")
    parser.add_argument("--keep_feedback_type", type=str, default='pos_neg', choices=["pos","neg", "", "pos_neg"],
                            help="Keep only pos or neg or both feedback for test.")
    parser.add_argument("--feedback_user_model", type=str, default='fa_fv', choices=["", "first", "random", "ra","rv", "ra_rv", "ra_fv", "fa_rv", "fa","fv", "fa_fv"],
                            help="Keep the first aspect, random aspect, first value, random value of the aspect.")
    parser.add_argument("--share_av_emb", action='store_true',
                            help="Separate or shared av embedding.")
    parser.add_argument("--sparse_emb", action='store_true',
                            help="use sparse embedding or not.")
    parser.add_argument("--scale_grad", action='store_true',
                            help="scale the grad of word and av embeddings.")
    parser.add_argument("--av_count_per_iter", type=int, default=1,
                            help="av pairs per iter.")
    parser.add_argument("--info_level", type=int, default=1,
                            help="different information levels users will provide.")
    parser.add_argument("-nw", "--weight_distort", action='store_true',
                            help="Set to True to use 0.75 power to redistribute for neg sampling .")
    parser.add_argument("--decode", action='store_true',
                            help="Set to True for testing.")
    parser.add_argument("--test_mode", type=str, default="product_scores",
            help="Test modes: product_scores -> output ranking results and ranking scores; (default is product_scores)")
    parser.add_argument("--rank_cutoff", type=int, default=100,
                            help="Rank cutoff for output ranklists.")
    parser.add_argument("--iter_count", type=int, default=5,
                            help="the number of round users interact with the system.")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    return parser.parse_args()

def get_model_path(args):
    model_path_arr = [args.model_net_struct, args.similarity_func]
    if args.model_net_struct == 'AVHEM':
        model_path_arr.append(args.comb_net_struct)
    model_path_arr += ["bz%d" % args.batch_size, args.optimizer]
    if args.L2_lambda > 0.:
        model_path_arr.append("%.0E" % Decimal(args.L2_lambda))
    model_path_arr += ["lr%.4f" % args.init_learning_rate, "subs%.0E" % Decimal(args.subsampling_rate)]
    if args.model_net_struct == 'AVHEM':
        model_path_arr.append(args.value_loss_func)
    model_path_arr += ["qw%.1f" % args.query_weight, "emb%d" % args.embed_size]
    if args.scale_grad:
        model_path_arr += ["scale%s" % args.scale_grad]
    if args.model_net_struct == 'AVHEM':
        model_path_arr += ["sparse%s%s" % (args.sparse_emb, args.loss_ablation), "lh%s.ckpt" % args.likelihood_way]
    else:
        model_path_arr += ["sparse%s.ckpt" % (args.sparse_emb)]
    model_path = "%s/model/%s" % (args.save_dir, "_".join(model_path_arr))
    print(model_path)
    return model_path

def create_model(args, data_set):
    """Create translation model and initialize or load parameters in session."""
    model = ProductSearchModel(
            data_set.vocab_size, data_set.user_size, data_set.product_size, data_set.query_max_length,
            data_set.word_dists, data_set.product_dists,
            data_set.aspect_dists, data_set.value_dists, data_set.av_id2word,
            data_set.aword_idx2_aid,
            data_set.aspect_keys, data_set.value_keys, data_set.aspect_value_count_dic,
            args.window_size, args.embed_size,
            args.query_weight, args.qnet_struct,
            args.comb_net_struct, args.model_net_struct,
            args.similarity_func, args.negative_sample,
            value_loss_func = args.value_loss_func,
            loss_ablation = args.loss_ablation,
            scale_grad = args.scale_grad,
            is_emb_sparse = args.sparse_emb,
            aspect_prob_type = args.aspect_prob_type,
            likelihood_way = args.likelihood_way,
            is_av_embed_shared = args.share_av_emb)
    if args.cuda:
        model = model.cuda()
    model.init_global_aspect_emb()
    if args.optimizer == 'adam':
        if args.sparse_emb:
            optimizer = torch.optim.SparseAdam(model.parameters(), lr = args.init_learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = args.init_learning_rate, weight_decay=args.L2_lambda)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.init_learning_rate, weight_decay=args.L2_lambda)
    model_path = get_model_path(args)

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch']
        args.init_learning_rate = checkpoint.get('learning_rate', args.init_learning_rate)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        print("Created model with fresh parameters.")
    return model, optimizer

def train(args):
    print("Reading data in %s" % args.data_dir)
    args.start_epoch = 0

    data_set = data_util.upsearch_data(
            args.data_dir, args.input_train_dir, 'train',
            args.batch_size, args.model_net_struct, args.threshold,
            is_av_embed_shared = args.share_av_emb,
            is_feedback_same_user = args.is_feedback_same_user,
            neg_per_pos = args.neg_per_pos)

    data_set.sub_sampling(args.subsampling_rate)
    orig_init_learning_rate = args.init_learning_rate
    learning_rate = orig_init_learning_rate
    model, optimizer = create_model(args, data_set)
    #args.init_learning_rate as current initial learning rate has been reset from the model loaded

    model.train()

    model_dir = "%s/model" % (args.save_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    words_to_train = float(args.max_train_epoch * data_set.word_count) + 1
    data_set.setup_data_set(words_to_train)
    previous_words, step_time, loss = 0.,0.,0.
    get_batch_time = 0.0
    start_time = time.time()
    current_epoch = args.start_epoch
    current_step = 0
    total_norm = 0.
    is_best = False
    while current_epoch < args.max_train_epoch:
        print("Initialize epoch:%d" % current_epoch)
        data_set.initialize_epoch()
        word_has_next = True

        while word_has_next:
            time_flag = time.time()
            batch_data = None
            #only when it's its turn and not empty
            #or when the other one is empty
            if word_has_next:
                #print("train word")
                if args.model_net_struct == 'AVHEM':
                    batch_data = data_set.get_av_train_batch()
                else:
                    batch_data = data_set.get_train_batch()
                word_has_next = batch_data[-1]

            get_batch_time += time.time() - time_flag
            pivot_time = time.time()
            learning_rate = args.init_learning_rate * max(0.0001,
                                1.0 - data_set.finished_word_num / words_to_train)
            if args.optimizer == 'sgd':
                adjust_learning_rate(optimizer, learning_rate)

            if len(batch_data[0]) > 0:
                time_flag = time.time()
                step_loss = model(batch_data[:-1])
                optimizer.zero_grad()
                step_loss.backward()
                cur_batch_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                total_norm += cur_batch_norm / args.steps_per_checkpoint
                optimizer.step()
                loss += step_loss.item() / args.steps_per_checkpoint #convert an tensor with dim 0 to value
                current_step += 1
                step_time += time.time() - time_flag

            # Once in a while, we print statistics.
            if current_step % args.steps_per_checkpoint == 0:
                print("Epoch %d Words %d/%d: lr = %5.4f loss = %6.2f words/sec = %5.2f prepare_time %.2f step_time %.2f" %
                        (current_epoch, data_set.finished_word_num, words_to_train, learning_rate, loss,
                            (data_set.finished_word_num - previous_words)/(time.time() - start_time), get_batch_time, step_time))#, end=""
                print("av_loss:%.2f ia_loss:%.2f iav_loss:%.2f iav_pos_loss:%.2f iav_neg_loss:%.2f up_loss1:%.2f" %
                        (model.group_av_loss / args.steps_per_checkpoint,
                            model.group_ia_loss / args.steps_per_checkpoint,
                            model.group_iav_loss / args.steps_per_checkpoint,
                            model.group_iav_pos_loss / args.steps_per_checkpoint,
                            model.group_iav_neg_loss / args.steps_per_checkpoint,
                            model.group_up_loss1 / args.steps_per_checkpoint))
                print("total_norm:%.2f word_loss:%.2f uw_loss:%.2f pw_loss:%.2f up_loss2:%.2f" %
                        (total_norm, model.group_word_loss / args.steps_per_checkpoint,
                            model.group_uw_loss / args.steps_per_checkpoint,
                            model.group_pw_loss / args.steps_per_checkpoint,
                            model.group_up_loss2 / args.steps_per_checkpoint))

                model.group_ia_loss, model.group_iav_loss, model.group_up_loss1 = 0., 0., 0.
                model.group_uw_loss, model.group_pw_loss, model.group_up_loss2 = 0., 0., 0.
                model.group_av_loss, model.group_word_loss, model.group_iav_pos_loss = 0., 0.,0.
                model.group_iav_neg_loss, step_time, get_batch_time = 0., 0.,0.

                total_norm, loss = 0., 0.
                current_step = 1
                sys.stdout.flush()
                previous_words = data_set.finished_word_num
                start_time = time.time()
        #save model after each epoch
        save_checkpoint({'epoch': current_epoch + 1,
            'learning_rate': learning_rate,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }, is_best, model_path)

        current_epoch += 1
    save_checkpoint({'epoch': current_epoch,
                    'learning_rate': learning_rate,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, is_best, model_path)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_product_scores(args):
    # Prepare data.
    print("Reading data in %s" % args.data_dir)

    ranklist_dir = "%s/ranklist" % (args.save_dir)
    if not os.path.isdir(ranklist_dir):
        os.makedirs(ranklist_dir)
    data_set = data_util.upsearch_data(
            args.data_dir, args.input_train_dir, 'test',
            args.batch_size, args.model_net_struct, args.threshold,
            is_av_embed_shared = args.share_av_emb)
    # Create model.
    orig_init_learning_rate = args.init_learning_rate
    model, optimizer = create_model(args, data_set)
    model.eval()
    #for dropout
    current_step = 0
    user_ranklist_map = colls.defaultdict(list)
    user_ranklist_score_map = colls.defaultdict(list)
    print('Start Testing')
    data_set.setup_data_set(0)
    data_set.initialize_test_iter()
    data_set.prepare_test_epoch()
    data_set.read_train_product_ids(args.input_train_dir)
    has_next = True
    while has_next:
        user_idxs, query_idxs, query_word_idxs, has_next = data_set.get_test_batch()
        if len(user_idxs) == 0:
            break
        #product_idxs can be refined to a candidate set

        user_product_scores = model.get_product_scores(user_idxs, query_word_idxs, product_idxs = None)
        current_step += 1
        #batch_size, product_size
        # record the results
        store_sorted_ranklist(data_set, args.rank_cutoff, \
            user_ranklist_map, user_ranklist_score_map,
            user_product_scores, user_idxs, query_idxs,
            args.keep_first_rel_qu, product_idxs = None)

        if current_step % args.steps_per_checkpoint == 0:
            print("Finish test qu %d/%d\r" %
                (data_set.cur_uqr_i, len(data_set.test_seq)), end="")

    fname_arr = [args.model_net_struct, args.similarity_func]
    fname_arr += ["bz%d" % args.batch_size, args.optimizer]
    if args.L2_lambda > 0.:
        fname_arr.append("%.0E" % Decimal(args.L2_lambda))
    fname_arr += ["lr%.4f" % orig_init_learning_rate, "subs%.0E" % Decimal(args.subsampling_rate)]
    fname_arr += ["qw%.1f" % args.query_weight, "emb%d" % args.embed_size]
    if args.scale_grad:
        fname_arr += ["scale%s" % args.scale_grad]
    fname_arr += ["sparse%s"% args.sparse_emb ,"firstkept%s.ckpt" % args.keep_first_rel_qu]
    fname = '_'.join(fname_arr)

    data_set.output_ranklist(user_ranklist_map, user_ranklist_score_map, ranklist_dir, fname)
    print("first relevant skipped qu:", data_set.skip_store_rank_count)

def get_product_scores_av(args):
    # Prepare data.
    print("Reading data in %s" % args.data_dir)

    data_set = data_util.upsearch_data(
            args.data_dir, args.input_train_dir, 'test',
            args.batch_size, args.model_net_struct, args.threshold,
            is_av_embed_shared = args.share_av_emb,
            is_feedback_same_user = args.is_feedback_same_user,
            neg_per_pos = args.neg_per_pos,
            keep_feedback_type = args.keep_feedback_type, #"pos", "neg"
            feedback_user_model = args.feedback_user_model, #ra, fa, rv, fv
            max_av_count = args.av_count_per_iter,
            info_level = args.info_level)
    #get q-u pair both with av and without av, the total number of which is args.batch_size
    # Create model.
    ranklist_dir = "%s/ranklist" % (args.save_dir)
    if not os.path.isdir(ranklist_dir):
        os.makedirs(ranklist_dir)
    orig_init_learning_rate = args.init_learning_rate
    model, optimizer = create_model(args, data_set)
    model.eval()
    model.init_aspect_vecs_for_test()
    #for dropout
    current_step = 0
    user_ranklist_map = colls.defaultdict(list)
    user_ranklist_score_map = colls.defaultdict(list)
    print('Start Testing')
    start_time = time.time()
    #data_set.setup_data_set(0)
    data_set.prepare_test_epoch()
    data_set.read_train_product_ids(args.input_train_dir)
    torch.set_grad_enabled(False)#do not require grad
    for cur_iter_i in xrange(1, args.iter_count + 1):
        data_set.initialize_test_iter()
        has_next = True
        qu_count = 0
        apaired_qu_count = 0
        while has_next:
            product_idxs = None
            av_product_idxs = None
            user_idxs, query_idxs, query_word_idxs, \
            av_user_idxs, av_query_idxs, av_query_word_idxs, query_av_pairs, has_next \
                = data_set.get_av_test_batch(user_ranklist_map, cur_iter_i)
            qu_count += len(user_idxs)
            apaired_qu_count += len(av_user_idxs)

            if len(user_idxs) == 0 and len(av_user_idxs) == 0:
                break
            if len(user_idxs) > 0:
                user_product_scores = model.get_product_scores(user_idxs, query_word_idxs, product_idxs = None)
                store_sorted_ranklist(data_set, args.rank_cutoff, \
                    user_ranklist_map, user_ranklist_score_map,
                    user_product_scores, user_idxs, query_idxs,
                    args.keep_first_rel_qu,
                    cur_iter = cur_iter_i, product_idxs = None)

            if len(av_user_idxs) > 0:
                user_av_product_scores = model.get_product_scores_av(
                        av_user_idxs, av_query_word_idxs,
                        query_av_pairs,
                        product_idxs = av_product_idxs)
                store_sorted_ranklist(data_set, args.rank_cutoff, \
                    user_ranklist_map, user_ranklist_score_map,
                    user_av_product_scores, av_user_idxs, av_query_idxs,
                    args.keep_first_rel_qu,
                    cur_iter = cur_iter_i, product_idxs = None)
                #print("len(av_user_idx)", len(av_user_idxs))
            current_step += 1
            #batch_size, product_size

            # record the results
            #if current_step % args.steps_per_checkpoint == 0:
            if current_step % 10 == 0:
                print("Finish test qu %d/%d" %
                    (data_set.cur_uqr_i, len(data_set.test_seq)), end="")
        print("Iter%d qu_count:%d apaired_qu_count:%d" % (cur_iter_i, qu_count, apaired_qu_count))

        fname_arr = [args.model_net_struct, args.similarity_func, args.comb_net_struct]
        fname_arr += ["bz%d" % args.batch_size, args.optimizer]
        if args.L2_lambda > 0.:
            fname_arr.append("%.0E" % Decimal(args.L2_lambda))
        fname_arr += ["lr%.4f" % orig_init_learning_rate, "subs%.0E" % Decimal(args.subsampling_rate)]
        fname_arr += ["a%s" % args.aspect_prob_type, args.value_loss_func]
        fname_arr += ["qw%.1f" % args.query_weight, "emb%d" % args.embed_size]
        if args.scale_grad:
            fname_arr += ["scale%s" % args.scale_grad]
        fname_arr += ["lh%s" % args.likelihood_way, "sparse%s%s" % (args.sparse_emb, args.loss_ablation)]
        fname_arr += ["info%d" % args.info_level, "sameu", str(args.is_feedback_same_user)]
        fname_arr += ["firstkept%s" % args.keep_first_rel_qu, args.keep_feedback_type]
        fname_arr += ["avcount%d" % args.av_count_per_iter]
        fname_arr += ["%siter%d.ckpt" % (args.feedback_user_model, cur_iter_i)]
        fname = '_'.join(fname_arr)

        data_set.output_ranklist(user_ranklist_map, user_ranklist_score_map, ranklist_dir, fname)
        print("first relevant skipped qu:", data_set.skip_store_rank_count)
        print("QU updated in this iter:", len(data_set.test_user_query_set))


def store_sorted_ranklist(data_set, cutoff,
                user_ranklist_map, user_ranklist_score_map,
                user_product_scores, user_idxs, query_idxs,
                keep_first_rel_qu,
                cur_iter = 1, product_idxs = None):
        #user_product_scores is a numpy array
    sorted_product_idxs = user_product_scores.argsort(axis=-1)[:,::-1] #by default axis=-1, along the last axis
    #the order is descending
    #print(sorted_product_idxs.shape)
    #batch_size, product_size,
    for i in xrange(user_product_scores.shape[0]):
        u_idx = user_idxs[i]
        q_idx = query_idxs[i]
        data_set.get_test_product_ranklist(
                        q_idx, u_idx, user_product_scores[i],
                        sorted_product_idxs[i], cur_iter, cutoff,
                        user_ranklist_map, user_ranklist_score_map,
                        keep_first_rel_qu) #(product name, rank)

def main(args):
    if args.decode:
        if args.test_mode == 'iter_result':
            #the first round does not use av vector.
            get_product_scores_av(args)
            #in the following iterations
        else:
            get_product_scores(args)
    else:
        train(args)
if __name__ == '__main__':
    main(parse_args())
