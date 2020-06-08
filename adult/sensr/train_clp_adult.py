import numpy as np
from sklearn.decomposition import TruncatedSVD
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import OrderedDict
import sys
import os
from sklearn.metrics import roc_auc_score

def get_consistency(X, l_pred, tf_X, relationship_idx = [33, 34, 35, 36, 37, 38], husband_idx = 33, wife_idx = 38):

    # make two copies of every datapoint: one which is a husband and one which is a wife. Then count how many classifications change
    X_husbands = np.copy(X)
    X_husbands[:,relationship_idx] = 0
    X_husbands[:,husband_idx] = 1

    husband_logits = l_pred.eval(feed_dict={tf_X: X_husbands})
    husband_preds = np.argmax(husband_logits, axis = 1)

    X_wives = np.copy(X)
    X_wives[:,relationship_idx] = 0
    X_wives[:,wife_idx] = 1

    wife_logits = l_pred.eval(feed_dict={tf_X: X_wives})
    wife_preds = np.argmax(wife_logits, axis = 1)

    spouse_consistency = np.mean([1 if husband_preds[i] == wife_preds[i] else 0 for i in range(len(husband_preds))])

    return spouse_consistency

def binary_accuracy_report(y_true, y_pred):
    
    idx_true_0 = np.where(y_true==0)[0]
    idx_true_1 = np.where(y_true==1)[0]
#    idx_pred_0 = np.where(y_pred==0)[0]
    idx_pred_1 = np.where(y_pred==1)[0]
#    print(y_true)
#    print(y_pred)
    acc = (y_true == y_pred).mean()
    
    tnr = (y_pred[idx_true_0]==0).mean()
    tpr = (y_pred[idx_true_1]==1).mean()
    
    bal_acc = (tnr + tpr)/2
    
    fdr = (y_true[idx_pred_1]==0).mean()
    
    return acc, bal_acc, fdr, tpr, tnr

def toxicity_report(logits, y, groups, group_names, prefix_mod, do_tb=True):
    
    summary_list=[]
    if do_tb:
        pred_probs = softmax(logits)
    else:
        pred_probs = logits
    auc_bal = roc_auc_score(y[:,1], pred_probs[:,1], average='weighted')
    auc = roc_auc_score(y[:,1], pred_probs[:,1])
    if do_tb:
        summary_list.append(tf.Summary.Value(tag=prefix_mod + ' AUC', simple_value = auc))
        summary_list.append(tf.Summary.Value(tag=prefix_mod + ' AUC balanced', simple_value = auc_bal))
        print(prefix_mod + ' AUC is %f; balanced AUC is %f' % (auc, auc_bal))
    preds = logits.argmax(axis=1)
    y_true = y.argmax(axis=1)
    acc, bal_acc, fdr, tpr, tnr = binary_accuracy_report(y_true, preds)
    append_summary(summary_list, [acc, bal_acc, fdr, tpr], prefix_mod, do_tb=do_tb)
    
    group_results = []
    all_gaps = []
    for i, g_n in enumerate(group_names):
        g_1_idx = np.where(groups[:,i])[0]
        acc_g, bal_acc_g, fdr_g, tpr_g_1, tnr_g_1 = binary_accuracy_report(y_true[g_1_idx], preds[g_1_idx])
        append_summary(summary_list, [acc_g, bal_acc_g, fdr_g, tpr_g_1], prefix_mod + ' ' + g_n, do_tb=do_tb)
        group_results.append([acc_g, bal_acc_g, fdr_g, tpr_g_1])
        g_0_idx = np.where(groups[:,i]==0)[0]
        _, _, _, tpr_g_0, tnr_g_0 = binary_accuracy_report(y_true[g_0_idx], preds[g_0_idx])
        all_gaps.append(tpr_g_1 - tpr_g_0)
        gaps_g = np.array([np.abs(tpr_g_1 - tpr_g_0), np.abs(tnr_g_1 - tnr_g_0)])
        gap_rms_g = np.sqrt((gaps_g**2).mean())
        print(prefix_mod + ' ' + g_n + ' gap RMS', gap_rms_g)
        if do_tb:
            summary_list.append(tf.Summary.Value(tag=prefix_mod + ' ' + g_n + ' gap RMS', simple_value = gap_rms_g))
            summary_list.append(tf.Summary.Value(tag=prefix_mod + ' ' + g_n + ' TPR gap', simple_value = tpr_g_1 - tpr_g_0))
    
    total_gap = np.sqrt((np.array(all_gaps)**2).mean())
    mean_gap = np.abs(all_gaps).mean()
    acc_var = np.std([g_r[0] for g_r in group_results])
    bal_acc_var = np.std([g_r[1] for g_r in group_results])
    fdr_var = np.std([g_r[2] for g_r in group_results])
    tpr_var = np.std([g_r[3] for g_r in group_results])
    
    if do_tb:
        summary_list.append(tf.Summary.Value(tag=prefix_mod + ' positives gap RMS ', simple_value = total_gap))
        summary_list.append(tf.Summary.Value(tag=prefix_mod + ' positives max gap', simple_value = np.abs(all_gaps).max())) 
        summary_list.append(tf.Summary.Value(tag=prefix_mod + ' positives mean absolute gap', simple_value = mean_gap))
        append_summary(summary_list, [acc_var, bal_acc_var, fdr_var, tpr_var], prefix_mod + ' STD', do_tb=do_tb)
        print(50*'-')
        return summary_list
    else:
        # return bal_acc, total_gap, np.abs(all_gaps).max()
        return bal_acc, bal_acc_var, acc_var
    
        
def append_summary(summary, vals, prefix, do_tb=True):
    acc, bal_acc, fdr, tpr = vals
    if do_tb:
        summary.append(tf.Summary.Value(tag=prefix + ' accuracy', simple_value = acc))
        summary.append(tf.Summary.Value(tag=prefix + ' balanced accuracy', simple_value = bal_acc))
        summary.append(tf.Summary.Value(tag=prefix + ' FDR', simple_value = fdr))
        summary.append(tf.Summary.Value(tag=prefix + ' TPR', simple_value = tpr))
    print(prefix + ' accuracy %.3f; balanced accuracy %.3f; FDR %.3f; TPR %.3f\n' % (acc, bal_acc, fdr, tpr))
    return
        

def bios_gap(logits, y, protected_y, y_names=None, protected_names=None, prefix='', do_tb=True, verbose=True):
    if protected_names is not None:
        p_0_name = protected_names[0]
        p_1_name = protected_names[1]
    else:
        p_0_name = '0'
        p_1_name = '1'
    C = y.shape[1]
    all_gaps = []
    all_tpr = []
    preds = logits.argmax(axis=1)
    
    summary_list=[]
                
    for c in range(C):
        if y_names is not None:
            c_name = y_names[c]
        else:
            c_name = str(c)
        idx_c = np.where(y[:,c])[0]
        if len(idx_c) < 10:
            print(prefix + 'Nothing in test for %s' % c_name)
            continue
        idx_0 = np.where(protected_y[idx_c,0])[0]
        idx_1 = np.where(protected_y[idx_c,1])[0]
        if len(idx_0) < 10:
            print(prefix + 'Nothing in test for %s %s' % (p_0_name, c_name))
            continue
        if len(idx_1) < 10:
            print(prefix + 'Nothing in test for %s %s' % (p_1_name, c_name))
            continue
        tpr_c = preds[idx_c]==c
        all_tpr.append(tpr_c.mean())
#        print(prefix + 'For class %s number of protected %s is %d; %s is %d' % (c_name, p_0_name, len(idx_0), p_1_name, len(idx_1)))
        tpr_0 = (preds[idx_c]==c)[idx_0]
        tpr_1 = (preds[idx_c]==c)[idx_1]
        gap_c = tpr_0.mean() - tpr_1.mean()
        all_gaps.append(gap_c)
        
        if verbose:
            print(prefix + 'For class %s TPR for protected %s is %.3f; %s is %.3f' % (c_name, p_0_name, tpr_0.mean(), p_1_name, tpr_1.mean()))
            print(prefix + 'Class %s gap is %.3f\n' % (c_name, gap_c))
        
        if do_tb:
            summary_list.append(tf.Summary.Value(tag=prefix + '%s TPR for %s' % (c_name, p_0_name), simple_value = tpr_0.mean()))
            summary_list.append(tf.Summary.Value(tag=prefix + '%s TPR for %s' % (c_name, p_1_name), simple_value = tpr_1.mean()))
            summary_list.append(tf.Summary.Value(tag=prefix + '%s gap' % c_name, simple_value = gap_c))
            
    total_gap = np.sqrt((np.array(all_gaps)**2).mean())
    mean_gap = np.abs(all_gaps).mean()
    if verbose:
        print(prefix + 'Gap RMS %.3f; mean abs gap %.3f; balanced TPR %.3f; max gap %.3f\n' % (total_gap, mean_gap, np.mean(all_tpr), np.abs(all_gaps).max()))
    if do_tb:
        summary_list.append(tf.Summary.Value(tag=prefix + 'gap RMS ', simple_value = total_gap))
        summary_list.append(tf.Summary.Value(tag=prefix + 'balanced TPR', simple_value = np.mean(all_tpr)))
        summary_list.append(tf.Summary.Value(tag=prefix + 'max gap', simple_value = np.abs(all_gaps).max())) 
        summary_list.append(tf.Summary.Value(tag=prefix + 'mean absolute gap', simple_value = mean_gap))
        return summary_list
    else:
        return total_gap, np.mean(all_tpr), np.abs(all_gaps).max(), mean_gap
    
    
def weight_variable(shape, name):
    if len(shape)>1:
        init_range = np.sqrt(6.0/(shape[-1]+shape[-2]))
    else:
        init_range = np.sqrt(6.0/(shape[0]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32) # seed=1000
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def compl_svd_projector(names, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(names)
        basis = tSVD.components_.T
        print('Singular values:')
        print(tSVD.singular_values_)
    else:
        basis = names.T    
    
    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl

def fc_network(variables, layer_in, n_layers, l=0, activ_f = tf.nn.relu, units = []):
    if l==n_layers-1:
        layer_out = tf.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)]
        units.append(layer_out)
        return layer_out, units
    else:
        layer_out = activ_f(tf.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)])
        l += 1
        units.append(layer_out)
        return fc_network(variables, layer_out, n_layers, l=l, activ_f=activ_f, units=units)
    
def forward(tf_X, tf_y, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=1e-6):
    
    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]
    else:
        n_features = int(tf_X.shape[1])
        n_class = int(tf_y.shape[1])
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]
        
    variables = OrderedDict()
    if weights is None:
        for l in range(n_layers):
            variables['weight_' + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(l))
            variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))
    else:
        weight_ind = 0
        for l in range(n_layers):
            variables['weight_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1
            variables['bias_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1


    ## Defining NN architecture
    l_pred, units = fc_network(variables, tf_X, n_layers, activ_f = activ_f)
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred))
            
    correct_prediction = tf.equal(tf.argmax(l_pred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if l2_reg > 0:
        loss = cross_entropy + l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
    else:
        loss = cross_entropy
        
    return variables, l_pred, loss, accuracy

def train_nn(X_train, y_train, X_test=None, y_test=None, weights=None, n_units = None, lr=0.001, batch_size=32, epoch=100, verbose=True, activ_f = tf.nn.relu, l2_reg=1e-6):
    N, D = X_train.shape
    
    try:
        K = y_train.shape[1]
    except:
        K = len(weights[-1])
    
    tf_X = tf.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')

    variables, l_pred, loss, accuracy = forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg)
    
    if epoch > 0:
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        n_per_class = int(batch_size/K)
        n_per_class = int(min(n_per_class, min(y_train.sum(axis=0))))
        batch_size = int(K*n_per_class)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for it in range(epoch):
            batch_idx = sample_batch_idx(y_train, n_per_class)
            
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            train_step.run(feed_dict={
                  tf_X: batch_x, tf_y: batch_y})
    
            if it % 10 == 0 and verbose:
                print('\nEpoch %d train accuracy %f' % (it, accuracy.eval(feed_dict={
                      tf_X: X_train, tf_y: y_train})))
                if y_test is not None:
                    print('Epoch %d test accuracy %g' % (it, accuracy.eval(feed_dict={
                          tf_X: X_test, tf_y: y_test})))
        if y_train is not None:
            print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_train, tf_y: y_train})))
        if y_test is not None:
            print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_test, tf_y: y_test})))
    
        weights = [x.eval() for x in variables.values()]
        train_logits = l_pred.eval(feed_dict={tf_X: X_train})
        if X_test is not None:
            test_logits = l_pred.eval(feed_dict={tf_X: X_test})
        else:
            test_logits = None
            
    return weights, train_logits, test_logits


def fair_dist(proj, w=0.1):
    tf_proj = tf.constant(proj, dtype=tf.float32)
    if w>0:
        return lambda x, y: tf.reduce_sum(tf.square(tf.matmul(x-y,tf_proj)) + w*tf.square(tf.matmul(x-y,tf.eye(proj.shape[0]) - tf_proj)), axis=1)
    else:
        return lambda x, y: tf.reduce_sum(tf.square(tf.matmul(x-y,tf_proj)), axis=1)

def explore_dist(sigma):
    tf_sigma = tf.constant(sigma, dtype=tf.float32)
    return lambda x,y: tf.reduce_sum(tf.matmul(x-y,tf_sigma)*(x-y), axis=1)

def softmax(logits):
    e_x = np.exp(logits - logits.max(axis=1).reshape(-1,1))
    return e_x / e_x.sum(axis=1).reshape(-1,1)

def predict_proba(X, weights):
    _, pred_logits, _ = train_nn(X, y_train=None, weights=weights, epoch=0)
    pred_probs = softmax(pred_logits)
    return pred_probs

def get_accuracy(logits, y):
    pred = logits.argmax(axis=1)
    true_y = y.argmax(axis=1)
    return (pred == true_y).mean()

def sample_batch_idx(y, n_per_class):
    batch_idx = []
    for i in range(y.shape[1]):
        batch_idx += np.random.choice(np.where(y[:,i]==1)[0], size=n_per_class, replace=False).tolist()
        
    np.random.shuffle(batch_idx)
    return batch_idx

def sample_balanced_batches(y, n_per_class, n_samples):
    batches = None
    for i in range(y.shape[1]):
        i_idx = np.random.choice(np.where(y[:,i]==1)[0], size=(n_samples,n_per_class))
        if batches is None:
            batches = i_idx
        else:
            batches = np.hstack((batches,i_idx))
        
    return batches

def forward_fair(tf_X, tf_y, tf_fair_X, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=1e-6):
    
    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]
    else:
        n_features = int(tf_X.shape[1])
        n_class = int(tf_y.shape[1])
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]
        
    variables = OrderedDict()
    if weights is None:
        for l in range(n_layers):
            variables['weight_' + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(l))
            variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))
    else:
        weight_ind = 0
        for l in range(n_layers):
            variables['weight_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1
            variables['bias_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1


    ## Defining NN architecture
    l_pred, units = fc_network(variables, tf_X, n_layers, activ_f = activ_f)
    l_pred_fair, units_fair = fc_network(variables, tf_fair_X, n_layers, activ_f = activ_f)
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred))
    cross_entropy_fair = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred_fair))
            
    correct_prediction = tf.equal(tf.argmax(l_pred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if l2_reg > 0:
        cross_entropy += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        cross_entropy_fair += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        
    return variables, l_pred, l_pred_fair, cross_entropy, cross_entropy_fair, accuracy

def forward_fair_clp(tf_X, tf_y, tf_fair_X, tf_counter_X, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=1e-6):
    
    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]
    else:
        n_features = int(tf_X.shape[1])
        n_class = int(tf_y.shape[1])
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]
        
    variables = OrderedDict()
    if weights is None:
        for l in range(n_layers):
            variables['weight_' + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(l))
            variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))
    else:
        weight_ind = 0
        for l in range(n_layers):
            variables['weight_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1
            variables['bias_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1


    ## Defining NN architecture
    l_pred, units = fc_network(variables, tf_X, n_layers, activ_f = activ_f)
    l_pred_fair, units_fair = fc_network(variables, tf_fair_X, n_layers, activ_f = activ_f)
    l_pred_counter, units_counter = fc_network(variables, tf_counter_X, n_layers, activ_f = activ_f)
    
    cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred)
    cross_entropy = tf.reduce_mean(cross_entropy_vector)
    cross_entropy_fair = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred_fair))
            
    correct_prediction = tf.equal(tf.argmax(l_pred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if l2_reg > 0:
        cross_entropy += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        cross_entropy_fair += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        
    return variables, l_pred, l_pred_fair, (cross_entropy, cross_entropy_vector), cross_entropy_fair, accuracy, l_pred_counter

COUNTER_INIT = 0.1
TB_BASE = './tensorboard_adult/'
RESULTS_BASE = './results/'

def train_fair_nn(X_train, y_train, tf_prefix='', X_test=None, X_test_counter=None, y_test=None, 
                  weights=None, n_units = None, balance_batch=True, lr=0.001, batch_size=100, epoch=100, 
                  verbose=True, activ_f = tf.nn.relu, l2_reg=0., plot=False, lamb_init=2., adv_epoch=100, 
                  adv_step=1., ro=None, fair_info=[], l2_attack=0.01, adv_epoch_full=10, lambda_clp=0.,
                  fair_start=0.5, counter_init=False, seed=None):
  
    groups_train, groups_test, group_names, protected_directions = fair_info
    K_protected = protected_directions.shape[0]
    proj_compl = compl_svd_projector(protected_directions, svd=-1)
    dist_f = fair_dist(proj_compl, 0.)
    
    global_step = tf.train.get_or_create_global_step()

    N, D = X_train.shape
    lamb = lamb_init
        
    try:
        K = y_train.shape[1]
    except:
        K = len(weights[-1])
    
    if balance_batch:
        n_per_class = int(batch_size/K)
        n_per_class = int(min(n_per_class, min(y_train.sum(axis=0))))
        batch_size = int(K*n_per_class)
        
    tf_X = tf.placeholder(tf.float32, shape=[None,D], name='tf_X')
    tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')
        
    ## Fair variables
    tf_counter_X = tf.placeholder(tf.float32, shape=[None,D], name='tf_counter_X')
    tf_directions = tf.constant(protected_directions, dtype=tf.float32)
    adv_weights = tf.Variable(tf.zeros([batch_size,K_protected]))
    
    full_adv_weights = tf.Variable(tf.zeros([batch_size,D]))
    
    if lambda_clp > 0:
        tf_fair_X = tf_counter_X + tf.matmul(adv_weights, tf_directions) + full_adv_weights
    else:
        tf_fair_X = tf_X + tf.matmul(adv_weights, tf_directions) + full_adv_weights
       
    variables, l_pred, l_pred_fair, (loss_clean, loss_clean_vector), loss_sensr, accuracy, l_pred_counter = forward_fair_clp(tf_X, tf_y, tf_fair_X, tf_counter_X, weights=weights, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg)

    if lambda_clp > 0:
        fair_subspace_loss = lambda_clp*tf.reduce_mean(tf.squared_difference(l_pred, l_pred_fair))
        train_loss = loss_clean + fair_subspace_loss
    else:
        fair_subspace_loss = loss_sensr
        train_loss = loss_sensr
        
    ## Attack is subspace
    fair_optimizer = tf.train.AdamOptimizer(learning_rate=adv_step)
#    fair_optimizer = tf.train.GradientDescentOptimizer(learning_rate=adv_step)
    
    fair_step = fair_optimizer.minimize(-fair_subspace_loss, var_list=[adv_weights], global_step=global_step)
    reset_fair_optimizer = tf.variables_initializer(fair_optimizer.variables())
    reset_adv_weights = adv_weights.assign(tf.zeros([batch_size,K_protected]))
    
    ## Attack out of subspace
    if lambda_clp > 0:
        distance = dist_f(tf_counter_X, tf_fair_X)
    else:
        distance = dist_f(tf_X, tf_fair_X)
        
    tf_lamb = tf.placeholder(tf.float32, shape=(), name='lambda')
    dist_loss = tf.reduce_mean(distance)
    fair_loss = fair_subspace_loss - tf_lamb*dist_loss
    
    tf_l2_attack = tf.placeholder(tf.float32, shape=(), name='full_attack_rate')
    if l2_attack > 0:
        full_fair_optimizer = tf.train.AdamOptimizer(learning_rate=tf_l2_attack)
#        full_fair_optimizer = tf.train.GradientDescentOptimizer(learning_rate=l2_attack)
        full_fair_step = full_fair_optimizer.minimize(-fair_loss, var_list=[full_adv_weights], global_step=global_step)
        reset_full_fair_optimizer = tf.variables_initializer(full_fair_optimizer.variables())
        reset_full_adv_weights = full_adv_weights.assign(tf.zeros([batch_size,D]))
        
    ## Train step
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_step = optimizer.minimize(train_loss, var_list=list(variables.values()), global_step=global_step)
    reset_optimizer = tf.variables_initializer(optimizer.variables())
    reset_main_step = True
    
    ######################
    
    failed_attack_count = 0
    failed_full_attack = 0
    failed_subspace_attack = 0
     
    tb_long = '_'.join(['fair-dim:' + str(K_protected), 'adv-epoch:' + str(adv_epoch), 
                        'batch_size:' + str(batch_size), 'adv-step:' + str(adv_step), 
                        'l2_attack:' + str(l2_attack), 'adv_epoch_full:' + str(adv_epoch_full), 
                        'ro:' + str(ro), 'balanced:' + str(balance_batch), 'lr:' + str(lr), 
                        'clp:' + str(lambda_clp), 'start:' + str(fair_start),
                        'c_init:' + str(counter_init)]) + '_' + 'arch:' + ','.join(list(map(str,n_units)))
    
    tb_base_dir = TB_BASE + tf_prefix + '_' + tb_long
    if seed is None:
        folder_exists = True
        post_idx = 0
        tb_dir = tb_base_dir + '_' + str(post_idx)
        while folder_exists:
            if os.path.exists(tb_dir):
                post_idx += 1
                tb_dir = tb_base_dir + '_' + str(post_idx)
            else:
                os.makedirs(tb_dir)
                folder_exists = False
    else:
        post_idx = seed
        tb_dir = tb_base_dir + '_' + str(post_idx)
        os.makedirs(tb_dir)
                     
    summary_writer = tf.summary.FileWriter(tb_dir)
    saver = tf.train.Saver(max_to_keep=3)
    out_freq = 1000
    save_freq = 10000
    fair_start = int(epoch*fair_start)
    counter_success_count = -1
    
    baseline_saved = False
    
    if balance_batch:
        batches = sample_balanced_batches(y_train, n_per_class, epoch)
    else:
        batches = np.random.choice(N, size=(epoch,batch_size))
        
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        for it in range(epoch):
            do_upd = True
            batch_x = X_train[batches[it]]
            batch_y = y_train[batches[it]]
            
            if it > fair_start:
                
                if lambda_clp>0.:
                    batch_flip_x = batch_x + np.matmul(np.random.uniform(-COUNTER_INIT, COUNTER_INIT, size=(batch_size,K_protected)),protected_directions)
                    
                if counter_init and lambda_clp<=0:
                    batch_loss_clean = loss_clean_vector.eval(feed_dict={tf_X: batch_x, tf_y: batch_y})
                    batch_counter_loss = loss_clean_vector.eval(feed_dict={tf_X: batch_flip_x, tf_y: batch_y})
                    batch_mask = (batch_counter_loss > batch_loss_clean).reshape(-1,1)
                    counter_success_count = batch_mask.sum()
                    batch_x = (1-batch_mask)*batch_x + batch_mask*batch_flip_x
                else:
                    counter_success_count = -1
    
                if reset_main_step:
                    sess.run(reset_optimizer)
                    reset_main_step = False
                
                if (not baseline_saved) and (fair_start>0):
                    print('Saving baseline before starting fair training')

                    try:
                        os.makedirs(RESULTS_BASE)
                    except:
                        pass
                    
                    saver.save(sess,
                             os.path.join(tb_dir, 'baseline_model'),
                             global_step=global_step)
                    
                    weights = [x.eval() for x in variables.values()]
                    np.save(RESULTS_BASE + tf_prefix + '_' + tb_long + '_' + 'baseline-weights' + '_' + str(post_idx), weights)
                        
                    print('Baseline train saved')
                    baseline_saved = True
                
                ## SenSR begins
                if lambda_clp > 0:
                    all_dict = {tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb, tf_l2_attack: l2_attack, tf_counter_X: batch_flip_x}
                    X_dict = {tf_X: batch_x, tf_counter_X: batch_flip_x}
                else:
                    all_dict = {tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb, tf_l2_attack: l2_attack}
                    X_dict = {tf_X: batch_x}
                    
                loss_before_subspace_attack = fair_loss.eval(feed_dict=all_dict)
                ## Do subspace attack
                for adv_it in range(adv_epoch):
                    fair_step.run(feed_dict=all_dict)
                ## Check result
                loss_after_subspace_attack = fair_loss.eval(feed_dict=all_dict)
                if loss_after_subspace_attack < loss_before_subspace_attack:
                        print('WARNING: subspace attack failed: objective decreased from %f to %f; resetting the attack' % (loss_before_subspace_attack, loss_after_subspace_attack))
                        sess.run(reset_adv_weights)
                        failed_subspace_attack += 1
                        
                if l2_attack > 0:
                    fair_loss_before_l2_attack = fair_loss.eval(feed_dict=all_dict)
    
                    ## Do full attack
                    for full_adv_it in range(adv_epoch_full):
                        full_fair_step.run(feed_dict=all_dict)
                    
                    ## Check result
                    fair_loss_after_l2_attack = fair_loss.eval(feed_dict=all_dict)
                    if fair_loss_after_l2_attack < fair_loss_before_l2_attack:
                        print('WARNING: full attack failed: objective decreased from %f to %f; skipping update steps' % (fair_loss_before_l2_attack, fair_loss_after_l2_attack))
                        failed_full_attack += 1
                        do_upd = False
                        l2_attack *= 0.999
                        print('Decreasing learning rate: new rate is %f' % l2_attack)
                    
                adv_batch = tf_fair_X.eval(feed_dict=X_dict)
                    
                if np.isnan(adv_batch.sum()):
                    print('Nans in adv_batch; making no change')
                    sess.run(reset_adv_weights)
                    if l2_attack > 0:
                        sess.run(reset_full_adv_weights)
                    failed_attack_count += 1
                    do_upd = False
                    
                elif ro is not None:
                    if do_upd:
                        mean_dist = dist_loss.eval(feed_dict=X_dict)                        
                        lamb = max(0.00001,lamb + (max(mean_dist,ro)/min(mean_dist,ro))*(mean_dist - ro))
                        lamb = min(lamb, 10)
            else:
                ## Baseline training
                adv_batch = batch_x
                if lambda_clp > 0:
                    all_dict = {tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb, tf_counter_X: batch_x}
                    X_dict = {tf_X: batch_x, tf_counter_X: batch_x}
                else:
                    all_dict = {tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb}
                    X_dict = {tf_X: batch_x}
            
            ## Parameter update step
            if do_upd:
                _, loss_at_update = sess.run([train_step,fair_loss], feed_dict=all_dict)
            else:
                loss_at_update = fair_loss.eval(feed_dict=all_dict)
            
            if it % out_freq == 0 and verbose:
                tf_dist = distance.eval(feed_dict=X_dict)
            if it > fair_start:
                sess.run(reset_adv_weights)
                sess.run(reset_fair_optimizer)
                if l2_attack > 0:
                    sess.run(reset_full_fair_optimizer)
                    sess.run(reset_full_adv_weights)
                    
            if (it % out_freq == 0 or it == epoch - 1) and verbose:
                dd = ((adv_batch-batch_x)**2).sum(axis=1)
                print('Real and fair distances (max/min/mean):')
                print(dd.max(), dd.min(), dd.mean())
                print(tf_dist.max(), tf_dist.min(), tf_dist.mean())
                print('Counter success count is %d' % counter_success_count)
                train_acc, train_logits, train_loss = sess.run([accuracy,l_pred,loss_clean], feed_dict={
                      tf_X: X_train, tf_y: y_train})
                print('Epoch %d train accuracy %f; loss %f; lambda is %f' % (it, train_acc, train_loss, lamb))
                if y_test is not None:
                    test_acc, test_logits = sess.run([accuracy,l_pred], feed_dict={
                            tf_X: X_test, tf_y: y_test})
                    print('Epoch %d test accuracy %g' % (it, test_acc))
                    if X_test_counter is not None:
                        probs_counter = []
                        for counter_X in X_test_counter:
                            counter_logits = l_pred.eval(feed_dict={tf_X: counter_X})
                            probs_counter.append(softmax(counter_logits)[:,1])
                        
                        probs_counter = np.column_stack(probs_counter)
                        consistency_score = probs_counter.std(axis=1).mean()
                        prediction_consistency = (probs_counter > 0.5).sum(axis=1)
                        prediction_consistency = ((prediction_consistency == 0) + (prediction_consistency == len(X_test_counter))).mean()
                        print('Epoch %d test consistency score is %g' % (it, consistency_score))
                        print('Epoch %d test prediction consistency is %g' % (it, prediction_consistency))
                        counter_summary = [tf.Summary.Value(tag='test consistency score', simple_value = consistency_score),
                                           tf.Summary.Value(tag='test prediction consistency', simple_value = prediction_consistency)]
                    else:
                        counter_summary = []
                        
                ## Debugging:
                if it > fair_start:
                    print('FAILED attacks: subspace %d; full %d; Nans after attack %d' % (failed_subspace_attack, failed_full_attack, failed_attack_count))
                    print('Loss clean %f; subspace %f; adv %f' % (loss_before_subspace_attack, loss_after_subspace_attack, loss_at_update))
                                    
                if plot:
                    spouse_consistency = get_consistency(X_test, l_pred, tf_X)
                    print('Epoch %d spouse consistency %g' % (it, spouse_consistency)) 
                    summary_train = toxicity_report(train_logits, y_train, groups_train, group_names, 'Train')
                    summary_test = toxicity_report(test_logits, y_test, groups_test, group_names, 'Test')

                    summary = tf.Summary(value=[
                    tf.Summary.Value(tag='spouse consistency', simple_value = spouse_consistency),
                    tf.Summary.Value(tag='train loss', simple_value = train_loss),
                    tf.Summary.Value(tag='lambda', simple_value = lamb),
                    tf.Summary.Value(tag='L2 max', simple_value = dd.max()),
                    tf.Summary.Value(tag='L2 mean', simple_value = dd.mean()),
                    tf.Summary.Value(tag='Fair distance max', simple_value = tf_dist.max()),
                    tf.Summary.Value(tag='Fair distance mean', simple_value = tf_dist.mean()),
                    tf.Summary.Value(tag='Distance mean difference', simple_value = dd.mean() - tf_dist.mean()),
                    tf.Summary.Value(tag='Distance max difference', simple_value = dd.max() - tf_dist.max())] +
                    summary_train + summary_test + counter_summary
                    )
                    summary_writer.add_summary(summary, it)
                    summary_writer.flush()
                
                sys.stdout.flush()
                
            if it % save_freq == 0:
                saver.save(sess,
                         os.path.join(tb_dir, 'fair_model'),
                         global_step=global_step)
        
        saver.save(sess,
                 os.path.join(tb_dir, 'fair_model'),
                 global_step=global_step)
                        
        if y_train is not None:
            print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_train, tf_y: y_train})))
        if y_test is not None:
            print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_test, tf_y: y_test})))
        if ro is not None:
            print('Final lambda %f' % lamb)
            
        weights = [x.eval() for x in variables.values()]
        try:
            os.makedirs(RESULTS_BASE)
        except:
            pass
        np.save(RESULTS_BASE + tf_prefix + '_' + tb_long + '_' + 'fair-weights' + '_' + str(post_idx), weights)
        
    return weights, train_logits, test_logits, lamb, variables
    
