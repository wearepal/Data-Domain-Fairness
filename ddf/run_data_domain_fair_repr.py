import re

import pandas as pd
import tensorflow as tf
import os.path
import fnmatch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import model_selection as cross_validation
import sys
import os

from ddf.datasets import load_dataset


def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0] // 2
    return tf.nn.top_k(v, m).values[m - 1]


def tf_median_pairwise_euclidean_distance(X):
    XX = tf.matmul(X, X, transpose_b=True)
    X_sqnorms = tf.diag_part(XX)
    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)
    pair_dist = (-2 * XX + c(X_sqnorms) + r(X_sqnorms))
    pair_dist = tf.nn.relu(pair_dist)
    sq_dist = tf.sqrt(pair_dist)
    med_sqrt = get_median(sq_dist)
    return med_sqrt


def pairwise_euclidean_distance(data):
    X, s, y = data

    XX = np.dot(X, X.T)
    X_sqnorms = np.diag(XX)
    r = lambda x: np.expand_dims(x, 0)
    c = lambda x: np.expand_dims(x, 1)
    pair_dist = (-2 * XX + c(X_sqnorms) + r(X_sqnorms))
    pair_dist[pair_dist < 0] = 0
    return np.sqrt(pair_dist)


def dense_bn_relu(inp, units, deploy):
    units = tf.layers.dense(
        inp, units, activation=tf.nn.relu,
        kernel_initializer=tf.uniform_unit_scaling_initializer(seed=888))
    return units


def quadratic_time_HSIC(data_first, data_second, sigma_first, sigma_second):
    XX = tf.matmul(data_first, data_first, transpose_b=True)
    YY = tf.matmul(data_second, data_second, transpose_b=True)
    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    gamma_first = 1.  # 1. / (2 * sigma_first**2) TODO
    gamma_second = 0.5  # 1. / (2 * sigma_second**2) TODO
    # use the second binomial formula
    Kernel_XX = tf.exp(-gamma_first * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
    Kernel_YY = tf.exp(-gamma_second * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    Kernel_XX_mean = tf.reduce_mean(Kernel_XX, 0, keep_dims=True)
    Kernel_YY_mean = tf.reduce_mean(Kernel_YY, 0, keep_dims=True)

    HK = Kernel_XX - Kernel_XX_mean
    HL = Kernel_YY - Kernel_YY_mean

    n = tf.cast(tf.shape(Kernel_YY)[0], tf.float32)
    HKf = HK / (n - 1)
    HLf = HL / (n - 1)

    # biased estimate
    hsic = tf.trace(tf.matmul(HKf, HLf))
    return hsic


def quadratic_time_MMD(data_first, data_second, data_third, data_fourth, sigma,
    data_fifth=None, data_sixth=None):

    # kernel width
    gamma = 1 / (2 * sigma**2)

    # handles X and S first
    XX_1 = tf.matmul(data_first, data_first, transpose_b=True)
    XX_2 = tf.matmul(data_third, data_third, transpose_b=True)

    YY_1 = tf.matmul(data_second, data_second, transpose_b=True)
    YY_2 = tf.matmul(data_fourth, data_fourth, transpose_b=True)

    X_12 = tf.matmul(data_first, data_third, transpose_b=True)
    Y_12 = tf.matmul(data_second, data_fourth, transpose_b=True)

    X_sqnorms_1 = tf.diag_part(XX_1)
    X_sqnorms_2 = tf.diag_part(XX_2)
    Y_sqnorms_1 = tf.diag_part(YY_1)
    Y_sqnorms_2 = tf.diag_part(YY_2)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    # use the second binomial formula
    Kernel_XX_1 = tf.exp(-gamma * (-2 * XX_1 + c(X_sqnorms_1) + r(X_sqnorms_1)))
    Kernel_XX_2 = tf.exp(-gamma * (-2 * XX_2 + c(X_sqnorms_2) + r(X_sqnorms_2)))

    Kernel_YY_1 = tf.exp(-gamma * (-2 * YY_1 + c(Y_sqnorms_1) + r(Y_sqnorms_1)))
    Kernel_YY_2 = tf.exp(-gamma * (-2 * YY_2 + c(Y_sqnorms_2) + r(Y_sqnorms_2)))

    Kernel_X_12 = tf.exp(-gamma * (-2 * X_12 + c(X_sqnorms_1) + r(X_sqnorms_2)))
    Kernel_Y_12 = tf.exp(-gamma * (-2 * Y_12 + c(Y_sqnorms_1) + r(Y_sqnorms_2)))

    # then handles the conditioning variable, Y
    if data_fifth==None:
        # use product kernels, a Hadamard product between the original kernel matrices for each variable
        Kernel_1 = tf.multiply(Kernel_XX_1,Kernel_YY_1)
        Kernel_2 = tf.multiply(Kernel_XX_2,Kernel_YY_2)
        Kernel_12 = tf.multiply(Kernel_X_12,Kernel_Y_12)

    else:
        # use product kernels, a Hadamard product between the original kernel matrices for each variable
        ZZ_1 = tf.matmul(data_fifth, data_fifth, transpose_b=True)
        ZZ_2 = tf.matmul(data_sixth, data_sixth, transpose_b=True)
        Z_12 = tf.matmul(data_fifth, data_sixth, transpose_b=True)

        Z_sqnorms_1 = tf.diag_part(ZZ_1)
        Z_sqnorms_2 = tf.diag_part(ZZ_2)

        Kernel_ZZ_1 = tf.exp(-gamma * (-2 * ZZ_1 + c(Z_sqnorms_1) + r(Z_sqnorms_1)))
        Kernel_ZZ_2 = tf.exp(-gamma * (-2 * ZZ_2 + c(Z_sqnorms_2) + r(Z_sqnorms_2)))
        Kernel_Z_12 = tf.exp(-gamma * (-2 * Z_12 + c(Z_sqnorms_1) + r(Z_sqnorms_2)))

        Kernel_1 = tf.multiply(Kernel_XX_1,Kernel_YY_1)
        Kernel_1 = tf.multiply(Kernel_1,Kernel_ZZ_1)

        Kernel_2 = tf.multiply(Kernel_XX_2,Kernel_YY_2)
        Kernel_2 = tf.multiply(Kernel_2,Kernel_ZZ_2)

        Kernel_12 = tf.multiply(Kernel_X_12,Kernel_Y_12)
        Kernel_12 = tf.multiply(Kernel_12,Kernel_Z_12)

    m = tf.cast(tf.shape(XX_1)[0],tf.float32)
    n = tf.cast(tf.shape(XX_2)[0],tf.float32)

    mmd2 = (tf.reduce_sum(Kernel_1) / (m * m)
          + tf.reduce_sum(Kernel_2) / (n * n)
          - 2 * tf.reduce_sum(Kernel_12) / (m * n))
    return 4.0*mmd2

def make_marginal_data(*arrays, random_seed=523423, model_config):
    # X is conditional independence of S given Y
    # X,S,Y; permute the S according to the Y

    np.random.seed(random_seed)
    # permute all of them once
    n_obs = len(arrays[0])
    rperm = np.random.permutation(n_obs)
    res = []
    for arr in arrays:
        res.append(arr[rperm,])

    # now, permute according to the conditional independency
    if model_config['equalized_odds']:
        # equalized odds
        df = pd.DataFrame(res[1])
        df = df.groupby(res[2].flatten(), group_keys=False).transform(np.random.permutation)
        res[1] = df.as_matrix()
    else:
        # equality of opportunity
        rperm = np.random.permutation(n_obs)
        res[1] = res[1][rperm,]

    return res


##### Performance Metric ######


def compute_accuracy_multi_pvalue(Y, predictions, Xcontrol):
    Xcontrol = [np.argmax(x) for x in Xcontrol]
    correct = np.sum(Y == predictions)
    acc = correct * 1. / Y.shape[0]
    acc_sensitive = np.zeros(np.unique(Xcontrol).shape[0])
    ii = 0
    for v in np.unique(Xcontrol):
        idx_ = Xcontrol == v
        acc_sensitive[ii] = np.sum(Y[idx_] == predictions[idx_,]) / (np.sum(idx_) * 1.)
        ii = ii + 1
    return acc, acc_sensitive  # , pvalue


def compute_accuracy_pvalue(Y, predictions, Xcontrol):
    correct = np.sum(Y == predictions)
    acc = correct * 1. / Y.shape[0]
    acc_sensitive = np.zeros(np.unique(Xcontrol).shape[0])
    ii = 0
    for v in np.unique(Xcontrol):
        idx_ = Xcontrol == v
        acc_sensitive[ii] = np.sum(Y[idx_] == predictions[idx_,]) / (np.sum(idx_) * 1.)
        ii = ii + 1
    return acc, acc_sensitive  # , pvalue


def compute_multi_fpr_fnr(Y, predictions, Xcontrol):
    Xcontrol = [np.argmax(x) for x in Xcontrol]
    fp = np.sum(np.logical_and(Y == 0.0, predictions == +1.0))  # something which is -ve but is misclassified as +ve
    fn = np.sum(np.logical_and(Y == +1.0, predictions == 0.0))  # something which is +ve but is misclassified as -ve
    tp = np.sum(
        np.logical_and(Y == +1.0, predictions == +1.0))  # something which is +ve AND is correctly classified as +ve
    tn = np.sum(
        np.logical_and(Y == 0.0, predictions == 0.0))  # something which is -ve AND is correctly classified as -ve
    fpr_all = np.float(fp) / np.float(fp + tn)
    fnr_all = np.float(fn) / np.float(fn + tp)
    tpr_all = np.float(tp) / np.float(tp + fn)
    tnr_all = np.float(tn) / np.float(tn + fp)

    fpr_fnr_tpr_sensitive = np.zeros((4, np.unique(Xcontrol).shape[0]))  # ~~~ I changed this from 3 so add tnr
    ii = 0
    for v in np.unique(Xcontrol):
        idx_ = Xcontrol == v
        fp = np.sum(np.logical_and(Y[idx_] == 0.0,
                                   predictions[idx_] == +1.0))  # something which is -ve but is misclassified as +ve
        fn = np.sum(np.logical_and(Y[idx_] == +1.0,
                                   predictions[idx_] == 0.0))  # something which is +ve but is misclassified as -ve
        tp = np.sum(np.logical_and(Y[idx_] == +1.0, predictions[
            idx_] == +1.0))  # something which is +ve AND is correctly classified as +ve
        tn = np.sum(np.logical_and(Y[idx_] == 0.0, predictions[
            idx_] == 0.0))  # something which is -ve AND is correctly classified as -ve
        fpr = np.float(fp) / np.float(fp + tn)
        fnr = np.float(fn) / np.float(fn + tp)
        tpr = np.float(tp) / np.float(tp + fn)
        tnr = np.float(tn) / np.float(tn + fp)
        fpr_fnr_tpr_sensitive[0, ii] = fpr
        fpr_fnr_tpr_sensitive[1, ii] = fnr
        fpr_fnr_tpr_sensitive[2, ii] = tpr
        fpr_fnr_tpr_sensitive[3, ii] = tnr
        ii = ii + 1
    return fpr_all, fnr_all, fpr_fnr_tpr_sensitive


def compute_fpr_fnr(Y, predictions, Xcontrol):
    fp = np.sum(np.logical_and(Y == 0.0, predictions == +1.0))  # something which is -ve but is misclassified as +ve
    fn = np.sum(np.logical_and(Y == +1.0, predictions == 0.0))  # something which is +ve but is misclassified as -ve
    tp = np.sum(
        np.logical_and(Y == +1.0, predictions == +1.0))  # something which is +ve AND is correctly classified as +ve
    tn = np.sum(
        np.logical_and(Y == 0.0, predictions == 0.0))  # something which is -ve AND is correctly classified as -ve
    fpr_all = np.float(fp) / np.float(fp + tn)
    fnr_all = np.float(fn) / np.float(fn + tp)
    tpr_all = np.float(tp) / np.float(tp + fn)
    tnr_all = np.float(tn) / np.float(tn + fp)

    fpr_fnr_tpr_sensitive = np.zeros((4, np.unique(Xcontrol).shape[0]))
    ii = 0
    for v in np.unique(Xcontrol):
        idx_ = Xcontrol == v
        fp = np.sum(np.logical_and(Y[idx_] == 0.0,
                                   predictions[idx_] == +1.0))  # something which is -ve but is misclassified as +ve
        fn = np.sum(np.logical_and(Y[idx_] == +1.0,
                                   predictions[idx_] == 0.0))  # something which is +ve but is misclassified as -ve
        tp = np.sum(np.logical_and(Y[idx_] == +1.0, predictions[
            idx_] == +1.0))  # something which is +ve AND is correctly classified as +ve
        tn = np.sum(np.logical_and(Y[idx_] == 0.0, predictions[
            idx_] == 0.0))  # something which is -ve AND is correctly classified as -ve
        fpr = np.float(fp) / np.float(fp + tn)
        fnr = np.float(fn) / np.float(fn + tp)
        tpr = np.float(tp) / np.float(tp + fn)
        tnr = np.float(tn) / np.float(tn + fp)
        fpr_fnr_tpr_sensitive[0, ii] = fpr
        fpr_fnr_tpr_sensitive[1, ii] = fnr
        fpr_fnr_tpr_sensitive[2, ii] = tpr
        fpr_fnr_tpr_sensitive[3, ii] = tnr
        ii = ii + 1
    return fpr_all, fnr_all, fpr_fnr_tpr_sensitive


######################
class Model:
    def __init__(self, features_size, protected_size, target_size, features_names, rff_map, rff_map_sens, to_deploy,
                 code_size,
                 encoder_hidden_sizes, decoder_hidden_sizes,
                 predictor_hidden_sizes,
                 hsic_cost_weight, pred_cost_weight, dec_cost_weight, rff_samples, equalized_odds, device):
        if not to_deploy:
            self.init_network(features_size, protected_size, target_size, features_names, code_size,
                              encoder_hidden_sizes, decoder_hidden_sizes,
                              predictor_hidden_sizes, to_deploy, device)
            self.init_training(hsic_cost_weight, pred_cost_weight, dec_cost_weight, rff_map, rff_map_sens,
                               equalized_odds)
            self.init_logging(hsic_cost_weight, pred_cost_weight, dec_cost_weight)
        else:
            self.init_network(features_size, protected_size, target_size, features_names, code_size,
                              encoder_hidden_sizes, decoder_hidden_sizes,
                              predictor_hidden_sizes, to_deploy, device)

    def init_network(self, features_size, protected_size, target_size, features,
                     code_size, encoder_hidden_sizes, decoder_hidden_sizes,
                     predictor_hidden_sizes, deploy, device):

        self.x = tf.placeholder(tf.float32, [None, features_size], name="x")
        self.s = tf.placeholder(tf.float32, [None, protected_size], name="s")
        self.y = tf.placeholder(tf.float32, [None, target_size], name="y")
        self.x_marg = tf.placeholder(tf.float32, [None, features_size], name="x_marg")
        self.s_marg = tf.placeholder(tf.float32, [None, protected_size], name="s_marg")
        self.y_marg = tf.placeholder(tf.float32, [None, target_size], name="y_marg")
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.device(device):
            with tf.variable_scope('encoder'):
                prev_layer = self.x
                for size in encoder_hidden_sizes:
                    prev_layer = dense_bn_relu(prev_layer, size, deploy)
                tmp = tf.layers.dense(
                    prev_layer, code_size, activation=None,
                    kernel_initializer=tf.uniform_unit_scaling_initializer(
                        seed=888))
                self.encoded = tf.nn.dropout(tmp, keep_prob=self.keep_prob, seed=888)

            with tf.variable_scope('decoder'):
                prev_layer = self.encoded
                for size in decoder_hidden_sizes:
                    prev_layer = dense_bn_relu(prev_layer, size, deploy)
                # take into account the structure of our features
                keys_f = features.keys()
                ii = 0
                for key_f in keys_f:
                    if sum(features[key_f]) > 0 and sum(features[key_f]) == 1:
                        inc_unit = tf.layers.dense(
                            prev_layer, sum(features[key_f]), activation=None,
                            kernel_initializer=tf.uniform_unit_scaling_initializer(seed=888))
                        if ii == 0:
                            self.decoded = inc_unit
                        else:
                            self.decoded = tf.concat([self.decoded, inc_unit], axis=1)
                    elif sum(features[key_f]) > 1:
                        inc_unit = tf.layers.dense(
                            prev_layer, sum(features[key_f]), activation=tf.nn.softmax,
                            kernel_initializer=tf.uniform_unit_scaling_initializer(seed=888))
                        if ii == 0:
                            if deploy:  # use one hot encoding at the deployment
                                self.decoded = tf.one_hot(tf.argmax(inc_unit, dimension=1), depth=sum(features[key_f]))
                            else:  # use soft outputs for learning
                                self.decoded = inc_unit
                        else:
                            if deploy:  # use one hot encoding at the deployment
                                self.decoded = tf.concat([self.decoded, tf.one_hot(tf.argmax(inc_unit, dimension=1),
                                                                                   depth=sum(features[key_f]))], axis=1)
                            else:  # use soft outputs for learning
                                self.decoded = tf.concat([self.decoded, inc_unit],
                                                         axis=1)
                    ii += 1

            with tf.variable_scope('encoder', reuse=True):
                prev_layer = self.x_marg
                for size in encoder_hidden_sizes:
                    prev_layer = dense_bn_relu(prev_layer, size, deploy)
                tmp = tf.layers.dense(
                    prev_layer, code_size, activation=None,
                    kernel_initializer=tf.uniform_unit_scaling_initializer(
                        seed=888))
                self.encoded_marginal = tf.nn.dropout(tmp, keep_prob=self.keep_prob, seed=888)

            with tf.variable_scope('decoder', reuse=True):
                prev_layer = self.encoded_marginal
                for size in decoder_hidden_sizes:
                    prev_layer = dense_bn_relu(prev_layer, size, deploy)
                # take into account the structure of our features
                keys_f = features.keys()
                ii = 0
                for key_f in keys_f:
                    if sum(features[key_f]) > 0 and sum(features[key_f]) == 1:
                        inc_unit = tf.layers.dense(
                            prev_layer, sum(features[key_f]), activation=None,
                            kernel_initializer=tf.uniform_unit_scaling_initializer(seed=888))
                        if ii == 0:
                            self.decoded_marginal = inc_unit
                        else:
                            self.decoded_marginal = tf.concat([self.decoded_marginal, inc_unit], axis=1)
                    elif sum(features[key_f]) > 1:
                        inc_unit = tf.layers.dense(
                            prev_layer, sum(features[key_f]), activation=tf.nn.softmax,
                            kernel_initializer=tf.uniform_unit_scaling_initializer(seed=888))
                        if ii == 0:
                            if deploy:  # use one hot encoding at the deployment
                                self.decoded_marginal = tf.one_hot(tf.argmax(inc_unit, dimension=1),
                                                                   depth=sum(features[key_f]))
                            else:  # use soft outputs for learning
                                self.decoded_marginal = inc_unit
                        else:
                            if deploy:  # use one hot encoding at the deployment
                                self.decoded_marginal = tf.concat([self.decoded_marginal,
                                                                   tf.one_hot(tf.argmax(inc_unit, dimension=1),
                                                                              depth=sum(features[key_f]))], axis=1)
                            else:  # use soft outputs for learning
                                self.decoded_marginal = tf.concat([self.decoded_marginal, inc_unit],
                                                                  axis=1)
                    ii += 1

            with tf.variable_scope('predictor'):
                prev_layer = self.decoded
                for size in predictor_hidden_sizes:
                    prev_layer = dense_bn_relu(prev_layer, size, deploy)
                self.y_logit = tf.layers.dense(
                    prev_layer, target_size, activation=None,
                    kernel_initializer=tf.uniform_unit_scaling_initializer(seed=888))
                self.y_prob = tf.nn.sigmoid(self.y_logit)
                self.y_pred = tf.cast(
                    tf.greater(self.y_prob, 0.5), tf.int32)

    def init_training(self, hsic_cost_weight, pred_cost_weight, dec_cost_weight,
                      rff_map, rff_map_sens, equalized_odds):

        self.y_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.y_logit, labels=tf.cast(self.y, tf.float32)))

        # compute MMD and decoder loss in the feature space
        # via random Fourier features machinery
        x_map = rff_map.map(self.x)
        x_marginal_map = rff_map.map(self.x_marg)
        decoded_map = rff_map.map(self.decoded)
        decoded_marginal_map = rff_map.map(self.decoded_marginal)
        sens_map = self.s
        sens_marginal_map = self.s_marg

        self.decoder_loss = tf.reduce_mean(
            tf.nn.l2_loss(self.x - self.decoded))

        if equalized_odds:
            # EQUALIZED ODDS
            print("Equalized Odds Criterion")
            self.cycling_cost = -quadratic_time_MMD(x_map - decoded_map,
                                                    sens_map, x_marginal_map - decoded_marginal_map,
                                                    sens_marginal_map, 0.2, self.y, self.y_marg)
        else:
            # EQUALITY of OPPORTUNITY
            print("Equal Opportunity Criterion")
            mask = tf.equal(tf.squeeze(self.y),1)
            x_map_pos = tf.gather_nd(x_map, tf.where(mask))
            x_marginal_map_pos = tf.gather_nd(x_marginal_map, tf.where(mask))
            decoded_map_pos = tf.gather_nd(decoded_map, tf.where(mask))
            decoded_marginal_map_pos = tf.gather_nd(decoded_marginal_map, tf.where(mask))
            sens_map_pos = tf.gather_nd(sens_map, tf.where(mask))
            sens_marginal_map_pos = tf.gather_nd(sens_marginal_map, tf.where(mask))

            self.cycling_cost = -(quadratic_time_HSIC(x_map_pos - decoded_map_pos, sens_map_pos, 1.0e1, 0.2))

            self.hsic_cost = quadratic_time_HSIC(decoded_map_pos, sens_map_pos, 1.0e1, 0.2)

        self.pred_loss = (hsic_cost_weight * self.cycling_cost +
                          (hsic_cost_weight * self.hsic_cost) +
                          dec_cost_weight * self.decoder_loss +
                          pred_cost_weight * self.y_cost)

        pred_vars = (
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder') +
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decoder') +
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'predictor')
        )
        self.train_pred = tf.train.AdamOptimizer().minimize(
            self.pred_loss, var_list=pred_vars)

    def init_logging(self, hsic_cost_weight, pred_cost_weight, dec_cost_weight):
        # which variables to log
        tf.summary.scalar("decoder_loss", self.decoder_loss)
        tf.summary.scalar("decoder_loss_with_weight", dec_cost_weight * self.decoder_loss)
        tf.summary.scalar("pred_loss", self.pred_loss)
        tf.summary.scalar("y_cost", self.y_cost)
        tf.summary.scalar("y_cost_with_weight", pred_cost_weight * self.y_cost)
        tf.summary.scalar("cycling_cost", self.cycling_cost)
        tf.summary.scalar("cycling_cost_with_weight", hsic_cost_weight * self.cycling_cost)
        tf.summary.scalar("hsic_cost", self.hsic_cost)
        tf.summary.scalar("hsic_cost_with_weight", hsic_cost_weight * self.hsic_cost)

        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False, dtype=tf.int32)
        self.inc_step = tf.assign(self.global_step, self.global_step + 1)

        self.global_iteration = tf.Variable(0, name='global_iteration',
                                            trainable=False, dtype=tf.int32)
        self.inc_iteration = tf.assign(self.global_iteration,
                                       self.global_iteration + 1)

    def fit(self, train_data, train_data_marginal, valid_data, valid_data_marginal, SEED_NUM, logs_dir, verbose,
            tf_config,
            n_iterations, batch_size, model_save_iterations, report_iterations,
            pred_steps_per_iteration,
            init_random_seed):

        X_train, s_train, y_train = train_data
        X_valid, s_valid, y_valid = valid_data

        X_train_marginal, s_train_marginal, y_train_marginal = train_data_marginal
        X_valid_marginal, s_valid_marginal, y_valid_marginal = valid_data_marginal

        model_saver = tf.train.Saver(max_to_keep=None)
        models_dir = logs_dir + '/models_{}/'.format(SEED_NUM)
        last_exists = True
        last_path = models_dir + 'last.session'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            last_exists = False

        with tf.Session(config=tf_config) as sess:
            np.random.seed(init_random_seed)
            tf.set_random_seed(0)
            if last_exists:
                model_saver.restore(sess, last_path)
            else:
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(logs_dir + '/tb/train',
                                                 sess.graph)
            valid_writer = tf.summary.FileWriter(logs_dir + '/tb/valid')

            total_batches = int(X_train.shape[0] // batch_size)

            def _train_feed_dict(step):
                begin = (step % total_batches) * batch_size
                end = (step % total_batches + 1) * batch_size
                return {
                    self.x: X_train[begin:end],
                    self.s: s_train[begin:end],
                    self.y: y_train[begin:end],
                    self.x_marg: X_train_marginal[begin:end],
                    self.s_marg: s_train_marginal[begin:end],
                    self.y_marg: y_train_marginal[begin:end],
                    self.keep_prob: 1.0
                }

            for _ in range(n_iterations):
                iteration = sess.run(self.inc_iteration)

                for _ in range(pred_steps_per_iteration):
                    step = sess.run(self.inc_step)
                    s, _ = sess.run([self.summary_op, self.train_pred],
                                    feed_dict=_train_feed_dict(step))
                    train_writer.add_summary(s, step)

                if iteration % report_iterations == 0:
                    s = sess.run(
                        self.summary_op,
                        feed_dict={self.x: X_valid,
                                   self.s: s_valid,
                                   self.y: y_valid,
                                   self.x_marg: X_valid_marginal,
                                   self.s_marg: s_valid_marginal,
                                   self.y_marg: y_valid_marginal,
                                   self.keep_prob: 1.0})
                    valid_writer.add_summary(s, step)

                if iteration % model_save_iterations == 0:
                    path = models_dir + 'iteration_{}.session'.format(iteration)
                    model_saver.save(sess, path)

                if verbose and iteration % report_iterations == 0:
                    print("Finished iteration {}".format(iteration))

            model_saver.save(sess, last_path)

    def predict(self, model, features, logs_dir_f, tf_config, iteration, SEED_NUM):
        model_saver = tf.train.Saver()
        with tf.Session(config=tf_config) as sess:
            path = '{}/models_{}/iteration_{}.session'.format(logs_dir_f, SEED_NUM, iteration)
            model_saver.restore(sess, path)
            y_pred = model.y_pred.eval({model.x: features, model.keep_prob: 1.0})
            y_prob = model.y_prob.eval({model.x: features, model.keep_prob: 1.0})
            decoded = model.decoded.eval({model.x: features, model.keep_prob: 1.0})
        return y_pred, y_prob, decoded


def train(data_train, data_train_marginal, data_valid, data_valid_marginal, x_size, s_size, y_size, med_sq_dist,
          features, logs_dir_f, SEED_NUM, model_config, fit_config, device="cpu"):
    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=x_size,
                                                                         output_dim=model_config['rff_samples'],
                                                                         stddev=med_sq_dist, seed=888,
                                                                         name='kernel_mapper')
    kernel_mapper_sens = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=s_size,
                                                                              output_dim=model_config['rff_samples'],
                                                                              stddev=1.0, seed=888,
                                                                              name='kernel_mapper_sens')

    if device == "cpu":
        device = "/cpu:0"
    else:
        device = "/gpu:0"

    tf.reset_default_graph()
    with tf.Graph().as_default():
        model = Model(features_size=x_size, protected_size=s_size, target_size=y_size, features_names=features,
                      rff_map=kernel_mapper, rff_map_sens=kernel_mapper_sens, to_deploy=False, device=device,
                      **model_config)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        model.fit(data_train, data_train_marginal, data_valid, data_valid_marginal, SEED_NUM,
                  tf_config=tf_config, verbose=True, logs_dir=logs_dir_f,
                  **fit_config)
    return True


def test(data_train, data_valid, data_test, features, logs_dir_f, SEED_NUM, model_config, device="cpu"):
    # Computational graphs are associated with Sessions. 
    # We should "clear out" the state of the Session so we don't have multiple placeholder objects floating around 
    # as we call save and restore()
    tf.reset_default_graph()

    if device == "cpu":
        device = "/cpu:0"
    else:
        device = "/gpu:0"

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    kernel_mapper = None
    kernel_mapper_sens = None
    model = Model(features_size=data_train[0].shape[1], protected_size=data_train[1].shape[1],
                  target_size=data_train[2].shape[1], features_names=features, rff_map=kernel_mapper,
                  rff_map_sens=kernel_mapper_sens, to_deploy=True, device=device,
                  **model_config)

    all_iterations = []
    for file in os.listdir(logs_dir_f + '/models_{}/'.format(SEED_NUM)):
        if fnmatch.fnmatch(file, 'iteration_*.session.meta'):
            iteration = file[len('iteration_'): -len('.session.meta')]
            all_iterations.append(int(iteration))
    all_iterations = sorted(all_iterations)

    X_train, s_train, y_train = data_train
    X_valid, s_valid, y_valid = data_valid
    X_test, s_test, y_test = data_test
    tpr_diff = []
    fpr_dif = []
    acc_ = []

    # perform classification here with X and Xtilde
    reg_array = [10 ** i for i in range(7)]
    n_splits = 3
    cv = cross_validation.StratifiedKFold(n_splits=n_splits, random_state=888, shuffle=True)
    # with Xtilde
    print("with Xtilde for all iterations")
    decoded_train = None
    decoded_test = None
    for iteration in all_iterations[-1:]:
        y_pred_train, y_prob_train, decoded_train = model.predict(model, X_train, logs_dir_f, tf_config, iteration,
                                                                  SEED_NUM)
        y_pred_valid, y_prob_valid, decoded_valid = model.predict(model, X_valid, logs_dir_f, tf_config, iteration,
                                                                  SEED_NUM)
        y_pred_test, y_prob_test, decoded_test = model.predict(model, X_test, logs_dir_f, tf_config, iteration,
                                                               SEED_NUM)

        cv_scores = np.zeros((len(reg_array), n_splits))
        for i, reg_const in enumerate(reg_array):
            cv_scores[i] = cross_validation.cross_val_score(
                svm.LinearSVC(C=reg_const, dual=False, tol=1e-6, random_state=888), decoded_train, y_train.flatten(),
                cv=cv)
        cv_mean = np.mean(cv_scores, axis=1)
        reg_best = reg_array[np.argmax(cv_mean)]
        print("Regularization ", reg_best)
        clf = svm.LinearSVC(C=reg_best, dual=False, tol=1e-6, random_state=888)
        clf.fit(decoded_train, y_train.flatten())

        predictions = clf.predict(decoded_test)
        # performance measurement
        acc, acc_sensitive = compute_accuracy_pvalue(y_test.flatten(), predictions,
                                                     s_test.flatten())
        print('SVM Accuracy: %.2f%%' % (acc * 100.))
        print("per sensitive value: %.2f, %.2f, (%.2f)" % (
        acc_sensitive[0] * 100., acc_sensitive[1] * 100., (acc_sensitive[0] - acc_sensitive[1]) * 100.))
        fpr, fnr, fpr_fnr_tpr_sensitive = compute_fpr_fnr(y_test.flatten(), predictions,
                                                          s_test.flatten())
        print('SVM FPR and FNR: %.2f, %.2f' % (fpr * 100., fnr * 100.))
        print("TPR per sensitive value: %.2f, %.2f, (%.2f)" % (
        fpr_fnr_tpr_sensitive[2, 0] * 100., fpr_fnr_tpr_sensitive[2, 1] * 100.,
        (fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.))
        print("FPR per sensitive value: %.2f, %.2f, (%.2f)" % (
        fpr_fnr_tpr_sensitive[0, 0] * 100., fpr_fnr_tpr_sensitive[0, 1] * 100.,
        (fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.))
        print("FNR per sensitive value: %.2f, %.2f, (%.2f)" % (
        fpr_fnr_tpr_sensitive[1, 0] * 100., fpr_fnr_tpr_sensitive[1, 1] * 100.,
        (fpr_fnr_tpr_sensitive[1, 0] - fpr_fnr_tpr_sensitive[1, 1]) * 100.))
        print("TNR per sensitive value: %.2f, %.2f, (%.2f)" % (
        fpr_fnr_tpr_sensitive[3, 0] * 100., fpr_fnr_tpr_sensitive[3, 1] * 100.,
        (fpr_fnr_tpr_sensitive[3, 0] - fpr_fnr_tpr_sensitive[3, 1]) * 100.))
        print("\n")
        acc_.append(acc)
        tpr_diff.append((fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.)
        fpr_dif.append((fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.)

        x_column_names = ["age",
                          "education-num",
                          "capital-gain",
                          "capital-loss",
                          "hours-per-week",
                          "workclass_Federal-gov",
                          "workclass_Local-gov",
                          "workclass_Never-worked",
                          "workclass_Private",
                          "workclass_Self-emp-inc",
                          "workclass_Self-emp-not-inc",
                          "workclass_State-gov",
                          "workclass_Without-pay",
                          "education_10th",
                          "education_11th",
                          "education_12th",
                          "education_1st-4th",
                          "education_5th-6th",
                          "education_7th-8th",
                          "education_9th",
                          "education_Assoc-acdm",
                          "education_Assoc-voc",
                          "education_Bachelors",
                          "education_Doctorate",
                          "education_HS-grad",
                          "education_Masters",
                          "education_Preschool",
                          "education_Prof-school",
                          "education_Some-college",
                          "marital-status_Divorced",
                          "marital-status_Married-AF-spouse",
                          "marital-status_Married-civ-spouse",
                          "marital-status_Married-spouse-absent",
                          "marital-status_Never-married",
                          "marital-status_Separated",
                          "marital-status_Widowed",
                          "occupation_Adm-clerical",
                          "occupation_Armed-Forces",
                          "occupation_Craft-repair",
                          "occupation_Exec-managerial",
                          "occupation_Farming-fishing",
                          "occupation_Handlers-cleaners",
                          "occupation_Machine-op-inspct",
                          "occupation_Other-service",
                          "occupation_Priv-house-serv",
                          "occupation_Prof-specialty",
                          "occupation_Protective-serv",
                          "occupation_Sales",
                          "occupation_Tech-support",
                          "occupation_Transport-moving",
                          "relationship_Husband",
                          "relationship_Not-in-family",
                          "relationship_Other-relative",
                          "relationship_Own-child",
                          "relationship_Unmarried",
                          "relationship_Wife",
                          # "sex_Female",
                          # "sex_Male",
                          "race_Amer-Indian-Eskimo",
                          "race_Asian-Pac-Islander",
                          "race_Black",
                          "race_Other",
                          "race_White",
                          "native-country_Cambodia",
                          "native-country_Canada",
                          "native-country_China",
                          "native-country_Columbia",
                          "native-country_Cuba",
                          "native-country_Dominican-Republic",
                          "native-country_Ecuador",
                          "native-country_El-Salvador",
                          "native-country_England",
                          "native-country_France",
                          "native-country_Germany",
                          "native-country_Greece",
                          "native-country_Guatemala",
                          "native-country_Haiti",
                          "native-country_Holand-Netherlands",
                          "native-country_Honduras",
                          "native-country_Hong",
                          "native-country_Hungary",
                          "native-country_India",
                          "native-country_Iran",
                          "native-country_Ireland",
                          "native-country_Italy",
                          "native-country_Jamaica",
                          "native-country_Japan",
                          "native-country_Laos",
                          "native-country_Mexico",
                          "native-country_Nicaragua",
                          "native-country_Outlying-US(Guam-USVI-etc)",
                          "native-country_Peru",
                          "native-country_Philippines",
                          "native-country_Poland",
                          "native-country_Portugal",
                          "native-country_Puerto-Rico",
                          "native-country_Scotland",
                          "native-country_South",
                          "native-country_Taiwan",
                          "native-country_Thailand",
                          "native-country_Trinadad&Tobago",
                          "native-country_United-States",
                          "native-country_Vietnam",
                          "native-country_Yugoslavia",
                          ]

        s_column_name = ["sensitive"]
        y_column_name = ["label"]

        train_x_dataframe = pd.DataFrame(X_train)
        train_s_dataframe = pd.DataFrame(s_train, dtype='int32')
        train_y_dataframe = pd.DataFrame(y_train, dtype='int32')

        train_x_dataframe.columns = x_column_names
        train_s_dataframe.columns = s_column_name
        train_y_dataframe.columns = y_column_name

        train_dataframe = pd.concat([train_x_dataframe, train_s_dataframe, train_y_dataframe], axis=1)

        train_x_tilde_dataframe = pd.DataFrame(decoded_train)
        train_x_tilde_dataframe.columns = x_column_names
        train_tilde_dataframe = pd.concat([train_x_tilde_dataframe, train_s_dataframe, train_y_dataframe], axis=1)

        train_dataframe.to_csv("seed_{}_stylingtrain_{}.csv".format(SEED_NUM, iteration), index=False)
        train_tilde_dataframe.to_csv("seed_{}_stylingtraintilde_{}.csv".format(SEED_NUM, iteration), index=False)

        test_x_dataframe = pd.DataFrame(X_test)
        test_s_dataframe = pd.DataFrame(s_test, dtype='int32')
        test_y_dataframe = pd.DataFrame(y_test, dtype='int32')

        test_x_dataframe.columns = x_column_names
        test_s_dataframe.columns = s_column_name
        test_y_dataframe.columns = y_column_name

        test_dataframe = pd.concat([test_x_dataframe, test_s_dataframe, test_y_dataframe], axis=1)

        test_x_tilde_dataframe = pd.DataFrame(decoded_test)
        test_x_tilde_dataframe.columns = x_column_names
        test_tilde_dataframe = pd.concat([test_x_tilde_dataframe, test_s_dataframe, test_y_dataframe], axis=1)

        test_dataframe.to_csv("seed_{}_stylingtest_{}.csv".format(SEED_NUM, iteration), index=False)
        test_tilde_dataframe.to_csv("seed_{}_stylingtesttilde_{}.csv".format(SEED_NUM, iteration), index=False)

    print(np.array(acc_))
    print(np.array(tpr_diff))
    print(np.array(fpr_dif))

    # performance measurement
    cv_scores = np.zeros((len(reg_array), n_splits))
    print("with X")
    for i, reg_const in enumerate(reg_array):
        cv_scores[i] = cross_validation.cross_val_score(
            svm.LinearSVC(C=reg_const, dual=False, tol=1e-6, random_state=888), X_train, y_train.flatten(), cv=cv)
    print("CV_Scores", cv_scores)
    cv_mean = np.mean(cv_scores, axis=1)
    print("CV Mean", cv_mean)
    reg_best = reg_array[np.argmax(cv_mean)]
    clf = svm.LinearSVC(C=reg_best, dual=False, tol=1e-6, random_state=888)
    clf.fit(X_train, y_train.flatten())
    predictions = clf.predict(X_test)

    acc, acc_sensitive = compute_accuracy_pvalue(y_test.flatten(), predictions,
                                                 s_test.flatten())
    print('SVM Accuracy: %.2f%%' % (acc * 100.))
    print('Reg: %.2f' % (reg_best))
    print("per sensitive value: %.2f, %.2f, (%.2f)" % (
    acc_sensitive[0] * 100., acc_sensitive[1] * 100., (acc_sensitive[0] - acc_sensitive[1]) * 100.))
    fpr, fnr, fpr_fnr_tpr_sensitive = compute_fpr_fnr(y_test.flatten(), predictions,
                                                      s_test.flatten())
    print('SVM FPR and FNR: %.2f, %.2f' % (fpr * 100., fnr * 100.))
    print("TPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[2, 0] * 100., fpr_fnr_tpr_sensitive[2, 1] * 100.,
    (fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.))
    print("FPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[0, 0] * 100., fpr_fnr_tpr_sensitive[0, 1] * 100.,
    (fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.))
    print("FNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[1, 0] * 100., fpr_fnr_tpr_sensitive[1, 1] * 100.,
    (fpr_fnr_tpr_sensitive[1, 0] - fpr_fnr_tpr_sensitive[1, 1]) * 100.))
    print("TNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[3, 0] * 100., fpr_fnr_tpr_sensitive[3, 1] * 100.,
    (fpr_fnr_tpr_sensitive[3, 0] - fpr_fnr_tpr_sensitive[3, 1]) * 100.))
    print("\n")
    # with Xtilde
    cv_scores = np.zeros((len(reg_array), n_splits))
    print("with Xtilde")
    for i, reg_const in enumerate(reg_array):
        cv_scores[i] = cross_validation.cross_val_score(
            svm.LinearSVC(C=reg_const, dual=False, tol=1e-6, random_state=888), decoded_train, y_train.flatten(), cv=cv)
    print("CV_Scores", cv_scores)
    cv_mean = np.mean(cv_scores, axis=1)
    print("CV Mean", cv_mean)
    reg_best = reg_array[np.argmax(cv_mean)]
    clf = svm.LinearSVC(C=reg_best, dual=False, tol=1e-6, random_state=888)
    clf.fit(decoded_train, y_train.flatten())
    predictions = clf.predict(decoded_test)
    # performance measurement
    acc, acc_sensitive = compute_accuracy_pvalue(y_test.flatten(), predictions,
                                                 s_test.flatten())
    print('SVM Accuracy: %.2f%%' % (acc * 100.))
    print("reg value: %.2f" % (reg_best))
    print("per sensitive value: %.2f, %.2f, (%.2f)" % (
    acc_sensitive[0] * 100., acc_sensitive[1] * 100., (acc_sensitive[0] - acc_sensitive[1]) * 100.))
    fpr, fnr, fpr_fnr_tpr_sensitive = compute_fpr_fnr(y_test.flatten(), predictions,
                                                      s_test.flatten())
    print('SVM FPR and FNR: %.2f, %.2f' % (fpr * 100., fnr * 100.))
    print("TPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[2, 0] * 100., fpr_fnr_tpr_sensitive[2, 1] * 100.,
    (fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.))
    print("FPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[0, 0] * 100., fpr_fnr_tpr_sensitive[0, 1] * 100.,
    (fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.))
    print("FNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[1, 0] * 100., fpr_fnr_tpr_sensitive[1, 1] * 100.,
    (fpr_fnr_tpr_sensitive[1, 0] - fpr_fnr_tpr_sensitive[1, 1]) * 100.))
    print("TNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[3, 0] * 100., fpr_fnr_tpr_sensitive[3, 1] * 100.,
    (fpr_fnr_tpr_sensitive[3, 0] - fpr_fnr_tpr_sensitive[3, 1]) * 100.))
    print("\n")

    acc, acc_sensitive = compute_accuracy_pvalue(y_test.flatten(), y_pred_test.flatten(), s_test.flatten())
    print('Encoder Accuracy: %.2f%%' % (acc * 100.))
    print("per sensitive value: %.2f, %.2f, (%.2f)" % (
    acc_sensitive[0] * 100., acc_sensitive[1] * 100., (acc_sensitive[0] - acc_sensitive[1]) * 100.))
    fpr, fnr, fpr_fnr_tpr_sensitive = compute_fpr_fnr(y_test.flatten(), y_pred_test.flatten(),
                                                      s_test.flatten())
    print('Encoder FPR and FNR: %.2f, %.2f' % (fpr * 100., fnr * 100.))
    print("TPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[2, 0] * 100., fpr_fnr_tpr_sensitive[2, 1] * 100.,
    (fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.))
    print("FPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[0, 0] * 100., fpr_fnr_tpr_sensitive[0, 1] * 100.,
    (fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.))
    print("FNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[1, 0] * 100., fpr_fnr_tpr_sensitive[1, 1] * 100.,
    (fpr_fnr_tpr_sensitive[1, 0] - fpr_fnr_tpr_sensitive[1, 1]) * 100.))
    print("TNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[3, 0] * 100., fpr_fnr_tpr_sensitive[3, 1] * 100.,
    (fpr_fnr_tpr_sensitive[3, 0] - fpr_fnr_tpr_sensitive[3, 1]) * 100.))

    return True


def delete_all_models(seed):
    models_dir = './models_{}/'.format(seed)
    if os.path.exists(models_dir):
        import shutil
        shutil.rmtree(models_dir)


def delete_files(seed):
    file_dir = '.'
    for f in os.listdir(file_dir):
        if re.search(r".*\.csv", f):
            os.remove(os.path.join(file_dir, f))


# To run call `python <this_filename> <int 0<=i<10>`
if __name__ == '__main__':
    print(sys.argv[1])
    SEED_NUM = int(sys.argv[1])

    DATASET_CONFIG = dict(
        train_path="data/preprocessed/adult/train.csv",
        test_path="data/preprocessed/adult/test.csv",
        data_name='adult',
        validation_size=2000,
        remake_test=True,
        test_size=15000
    )

    MODEL_CONFIG = dict(
        code_size=40,
        encoder_hidden_sizes=[40],
        decoder_hidden_sizes=[40],
        predictor_hidden_sizes=[],
        hsic_cost_weight=100.,
        pred_cost_weight=1.,
        dec_cost_weight=1.0e-4,
        rff_samples=2000,
        equalized_odds=False
    )

    FIT_CONFIG = dict(
        n_iterations=50000,
        batch_size=2000,
        model_save_iterations=50000,
        report_iterations=1000,
        pred_steps_per_iteration=1,
        init_random_seed=888
    )

    RANDOM_SEEDS = [87656123, 741246123, 292461935, 502217591, 9327935, 2147631, 2010588, 5171154, 6624906, 5136170]

    FEATURE_SPLITS = ['sex_salary']

    s_scaler = None
    i_scaler = MinMaxScaler

    data_train, data_valid, data_test, features = load_dataset(
        random_state=RANDOM_SEEDS[SEED_NUM], feature_split=FEATURE_SPLITS[0],
        input_scaler=i_scaler,
        sensitive_scaler=s_scaler, **DATASET_CONFIG)

    x_size = data_train[0].shape[1]
    s_size = data_train[1].shape[1]
    y_size = data_train[2].shape[1]
    assert y_size == 1, "Target must be a single feature."

    sq_dist = pairwise_euclidean_distance(data_train)
    med_sq_dist = 1.4  # np.median(sq_dist)/2.
    # print("median of sqrt pairwise distance", med_sq_dist)

    logs_dir_f = "."

    data_train_marginal = make_marginal_data(*data_train, random_seed=996723)
    data_valid_marginal = make_marginal_data(*data_valid, random_seed=742134)

    train(data_train, data_train_marginal, data_valid, data_valid_marginal, x_size, s_size, y_size, med_sq_dist,
          features, logs_dir_f, SEED_NUM)
    test(data_train, data_valid, data_test, features, logs_dir_f, SEED_NUM)
