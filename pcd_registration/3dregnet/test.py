import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import time
import sklearn.metrics as metrics
import pickle
from registration.registration import initializePointCoud, globalRegistration, selectFunction 
from registration.registration import refine_registration, makeMatchesSet, draw_reg_result
import copy
import open3d as o3d

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source, target])


#######################################################
def saveToFile(dict):
    f = open('results/test.pickle', 'wb')
    pickle.dump(dict, f)
    f.close()


#######################################################
def RotationError(R1, R2):
    R1 = np.real(R1)
    R2 = np.real(R2)
    R_ = np.matmul(R1, np.linalg.inv(R2))
    ae = np.arccos((np.trace(R_) - 1) / 2)
    ae = np.rad2deg(ae)   
    frob_norm = np.linalg.norm(R_ - np.eye(3), ord='fro')
    return ae, frob_norm


#######################################################
def TranslationError(t1, t2):
    return np.linalg.norm(t1-t2)


#######################################################s
def test_process(mode, sess, data, cur_global_step, summary_writer, config, xin1, xin2, Rin, 
                 tin, fin, is_training, Rhat, that, logits, loss, weights, reg_flag, reg_function):
    print("Start testing...\n")
    x1 = data['x1']
    x2 = data['x2']
    pts1 = data['sd']
    pts2 = data['td']
    Rs = data['R']
    ts = data['t']
    fs = data['flag']
    R_hats = []
    t_hats = []
    flags = []
    delta = []
    cls_logs = []
    cls_weights = []
    losses = []
    accuracies = []
    per = 1
    refine = True

    x1_b = np.array(x1.data.reshape(1, 1, -1, 3))
    x1_b = x1_b[:, :, :int(x1_b.shape[2] * per), :]
    x2_b = np.array(x2.data.reshape(1, 1, -1, 3))
    x2_b = x2_b[:, :, :int(x2_b.shape[2] * per), :]
    Rs_b = np.array(Rs.data.reshape(1, 9))
    ts_b = np.array(ts.data.reshape(1, 3))     
    fs_b = np.array(fs.data.reshape(1,-1))
    fs_b = fs_b[:, :int(fs_b.shape[1] * per)]

    feed_dict = {
        xin1: x1_b,
        xin2: x2_b,
        Rin: Rs_b,
        tin: ts_b,
        fin: fs_b,
        is_training: False,
    }

    fetch = {
        "R_hat": Rhat,
        "t_hat": that,
        "logits": logits,
        "weights": weights,
        "loss": loss,
    }

    time_start = time.time()
    res = sess.run(fetch, feed_dict=feed_dict)
    time_end = time.time() - time_start

    delta.append(time_end)
    R_hats.append(res['R_hat'])
    t_hats.append(res['t_hat'])
    cls_logs.append(res['logits'])
    cls_weights.append(res['weights'])
    losses.append(res['loss'])
    log_pos = np.array(res['logits'] > 0, dtype=np.uint8)
    flags.append(log_pos.astype(bool))
    accuracies.append(metrics.accuracy_score(fs_b[0], log_pos[0]))

    t_errors = []
    r_errors = []
    frob_errors = []
    els = []

    if reg_flag:
        reg_fun = selectFunction(reg_function)
        if np.asarray(pts1.points).size > 0:
            result, elapsed, pts1_down, pts2_down = globalRegistration(pts1, pts2, reg_fun)
            print('{} registration took: {}'.format(reg_function, elapsed))
            delta = elapsed
            e = elapsed
            if reg_function == 'global' and refine:
                print()
                transformation, elapsed = refine_registration(pts1_down, pts2_down, result.correspondence_set)
                print('{} refinement took: {}'.format('Umeyama', elapsed))
                delta = delta + elapsed
                elapsed = e + elapsed
                els.append(elapsed)
                R_hat = transformation[:3,:3]
                t_hat = transformation[:3, 3]
            else:
                els.append(elapsed)
                R_hat = result.transformation[:3, :3]
        else:
            print('Point cloud registered')
            R_hat = R_hats[0].reshape(3, 3)
    else:
        if refine:
            transformation = np.eye(4)
            transformation[:3, :3] = R_hats[0].reshape(3, 3)
            transformation[:3,  3] = t_hats[0]
            set = makeMatchesSet(flags[0])
            transformation, elapsed = refine_registration(pts1, pts2, set)
            print('{} refinement took: {}'.format('Umeyama', elapsed))               
            delta = delta + elapsed
            els.append(elapsed)
            R_hat = transformation[:3, :3]
        else:
            R_hat = R_hats[0].reshape(3, 3)
            if config["representation"] == 'linear':
                u, s, vh = np.linalg.svd(R_hat, compute_uv=True)
                R_hat = np.matmul(u, vh)
                R_hat = np.linalg.det(R_hat) * R_hat
    distance_threshold = 0.05 * 0.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        pts1, pts2, x1, x2, o3d.pipelines.registration.FastGlobalRegistrationOption(0.05 * 0.5))
    draw_registration_result(pts1, pts2, result.transformation)

    R_gt = Rs[0].reshape(3,3)
    t_gt = ts[0]
    rot_error, frob_error = RotationError(R_gt, R_hat)
    r_errors.append(rot_error)
    frob_errors.append(frob_error)
    t_errors.append(TranslationError(t_hat, t_gt))
       
    mean_rotation_error = np.mean(np.abs(r_errors))
    mean_translation_error = np.mean(np.abs(t_errors))
    median_rotation_error = np.median(np.abs(r_errors))
    median_translation_error = np.median(np.abs(t_errors))
    mean_frob_error = np.median(np.abs(frob_errors))
    mean_losses = np.mean(np.abs(losses))
    mean_accuracies = np.mean(accuracies)
    mean_deltas = np.mean(delta)

    indx_min = np.argmin(np.abs(r_errors))
    summaries = []

    summaries += [
        tf.compat.v1.Summary.Value(tag="ErrorComputation/mean_rotation_error_{}".format(mode),
            simple_value=mean_rotation_error)]
    summaries += [
        tf.compat.v1.Summary.Value(tag="ErrorComputation/mean_translation_error_{}".format(mode),
            simple_value=mean_translation_error)]
    summaries += [
        tf.compat.v1.Summary.Value(tag="ErrorComputation/mean_frob_error_{}".format(mode),
            simple_value=mean_frob_error)]
    summaries += [
        tf.compat.v1.Summary.Value(tag="ErrorComputation/mean_losses_{}".format(mode),
            simple_value=mean_losses)]
    summaries += [
        tf.compat.v1.Summary.Value(tag="ErrorComputation/mean_accuracies_{}".format(mode),
                         simple_value=mean_accuracies)]
    summary_writer.add_summary(
        tf.compat.v1.Summary(value=summaries), global_step=cur_global_step)
        
    if els:
        print('Mean elapsed - {}'.format(np.mean(els)))
        
    print('Idx min: {} \tRot_min = {}'.format(indx_min, np.abs(r_errors[indx_min])))

    dict = {'RotationError':np.abs(r_errors),'TranslationError': np.abs(t_errors), 'FrobError': frob_errors,
            'TotalLoss': losses, 'Accuracies': accuracies, 'Delta': delta, 'R_gt': Rs, 't_gt': ts, 'x1': x1,
            'x2': x2, 'flag':flags, 'Rhat': R_hats, 'that': t_hats}
    # saveToFile(dict)

    return mean_frob_error





