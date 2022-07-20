import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()
from ops import tf_matrix_vector_mul, tf_add_vectors, tf_matrix_from_quaternion
from test import test_process

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/libroyale/python/pcd_registration/3Dregnet/'

#######################################################
class Network(object):

    #######################################################
    def __init__(self, config):
        self.config = config
        gpu_flag = False
        if config["gpu_options"] == 'gpu':
            device_name = '/' + config["gpu_options"] + ':' + config["gpu_number"]
            gpu_flag = True
        else:
            device_name = '/' + config["gpu_options"] + ':0'
        self._init_tensorflow(gpu_flag)
        with tf.device(device_name):
            self._build_placeholder()
            self._build_loss_func()
            self._build_model()
            self._build_loss()
            self._build_summary()
            self._build_optim()
            self._build_writer()

    #######################################################
    def _init_tensorflow(self, gpu_flag):
        if not gpu_flag:
            num_threads = (int)(os.popen('sysctl -n hw.ncpu').read())
            if num_threads != "":
                num_threads = int(num_threads)
                # print("Limiting tensorflow to {} threads.".format(num_threads))
                tfconfig = tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=num_threads,
                    inter_op_parallelism_threads=num_threads)
            else:
                tfconfig = tf.compat.v1.ConfigProto()
        else:
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.compat.v1.Session(config=tfconfig)

    #######################################################
    def _build_placeholder(self):
        self.x_in_1 = tf.compat.v1.placeholder(tf.float32, [None, 1, None, 3], name="x_in1")
        self.x_in_2 = tf.compat.v1.placeholder(tf.float32, [None, 1, None, 3], name="x_in2")
        self.x_in = tf.concat([self.x_in_1, self.x_in_2], 3)

        self.R_in = tf.compat.v1.placeholder(tf.float32, [None, 9], name="R_in")
        self.t_in = tf.compat.v1.placeholder(tf.float32, [None, 3], name="t_in")
        self.f_in = tf.compat.v1.placeholder(tf.float32, [None, None], name="f_in")
        
        self.is_training = tf.compat.v1.placeholder(tf.bool, (), name="is_training")

        # Global step for optimization
        self.global_step = tf.compat.v1.get_variable("global_step", shape=(),
                                                    initializer=tf.zeros_initializer(),
                                                    dtype=tf.int64,
                                                    trainable=False)

    #######################################################
    def _build_loss_func(self):
        from ops import l1, l2, geman_mcclure, l05
        # print('Loss Function Selected - {}'.format(self.config["loss_function"]))
        if self.config["loss_function"] == 'l1':
            self.loss_function = l1
        elif self.config["loss_function"] == 'l2' or self.config["loss_function"] == 'wls':
            self.loss_function = l2
        elif self.config["loss_function"] == 'gm':
            self.loss_function = geman_mcclure
        elif self.config["loss_function"] == 'l05':
            self.loss_function = l05

    #######################################################
    def _build_model(self):
        # x_shp = tf.shape(self.x_in)           # For determining the runtime shape
        from archs.arch import build_graph      # Network architecture 
        # print("Building Graph")
        self.logits, self.R_hat, self.t_hat = build_graph(self.x_in, self.is_training, self.config)
        self.weights = tf.nn.relu(tf.tanh(self.logits))

        from ops import tf_skew_symmetric
        if self.config["representation"] == 'lie':
            self.skew = tf_skew_symmetric(self.R_hat)
            self.R_hat = tf.reshape(self.skew, [-1, 3, 3])
            self.R_hat = tf.linalg.expm(self.R_hat)
        elif self.config["representation"] == 'quat':
            self.R_hat = tf_matrix_from_quaternion(self.R_hat)
            self.R_hat = tf.reshape(self.R_hat, [-1, 3, 3])
        elif self.config["representation"] == 'linear':
            self.R_hat = tf.reshape(self.R_hat, [-1, 3, 3])
        else:
            print('Not a valid representation')
            exit(10)

    #######################################################
    def _build_loss(self):
        with tf.compat.v1.variable_scope("Loss", reuse=tf.compat.v1.AUTO_REUSE):
            # sh = tf.shape(self.x_in_2)
            x1_ = tf.squeeze(self.x_in_1, axis=1)
            x2_ = tf.squeeze(self.x_in_2, axis=1)

            # print(self.R_hat.shape)
            self.x2_hat = tf_matrix_vector_mul(self.R_hat, x1_)
            self.x2_hat = tf_add_vectors(self.x2_hat, self.t_hat)
            sub = self.x2_hat - x2_

            if self.config["loss_function"] == 'wls':
                w_mul = tf.expand_dims(self.weights, axis=2)
                w_mul1 = tf.tile(w_mul, [1, 1, 3])
                sub = w_mul1*sub

            # print(sub.shape)
            r_loss = self.loss_function(sub)
            r_loss = tf.reduce_sum(r_loss, axis=1)

            self.rec_loss = tf.reduce_mean(r_loss)
            self.rec_loss = (self.config["loss_reconstruction"] * self.rec_loss * 
                             tf.compat.v1.cast(self.global_step >= 
                             tf.compat.v1.cast(self.config["loss_reconstruction_init_iter"], 
                             dtype=tf.int64), dtype=tf.float32))

            tf.summary.scalar("reconstruction_loss", self.rec_loss)

            is_pos = tf.compat.v1.cast(self.f_in > 0, dtype=tf.float32)
            is_neg = tf.compat.v1.cast(self.f_in <= 0, dtype=tf.float32)
            c = is_pos - is_neg

            clf_losses = -tf.compat.v1.log(tf.nn.sigmoid(c * self.logits))
            num_pos = tf.nn.relu(tf.reduce_sum(is_pos, axis=1) - 1.0) + 1.0
            num_neg = tf.nn.relu(tf.reduce_sum(is_neg, axis=1) - 1.0) + 1.0
            classif_loss_p = tf.reduce_sum(clf_losses * is_pos, axis=1)
            classif_loss_n = tf.reduce_sum(clf_losses * is_neg, axis=1)
            self.clf_loss = tf.reduce_mean(classif_loss_p * 0.5 / num_pos +
                                           classif_loss_n * 0.5 / num_neg)
            self.clf_loss = self.config["loss_classif"] * self.clf_loss
            tf.summary.scalar("classification_loss", self.clf_loss)
            tf.summary.scalar("classif_loss_p", tf.reduce_mean(classif_loss_p * 0.5 / num_pos))
            tf.summary.scalar("classif_loss_n", tf.reduce_mean(classif_loss_n * 0.5 / num_neg))

            for var in tf.compat.v1.trainable_variables():
                if "weights" in var.name:
                    tf.compat.v1.add_to_collection("l2_losses", tf.reduce_sum(var ** 2))
            l2_loss = tf.add_n(tf.compat.v1.get_collection("l2_losses"))
            tf.summary.scalar("l2_loss", l2_loss)

            self.loss = self.config["loss_decay"] * l2_loss
            self.loss += self.rec_loss
            self.loss += self.clf_loss

            tf.summary.scalar("loss", self.loss)

    #######################################################
    def _build_summary(self):
        self.summary_op = tf.compat.v1.summary.merge_all()

    #######################################################
    def _build_optim(self):
        with tf.compat.v1.variable_scope("Optimization", reuse=tf.compat.v1.AUTO_REUSE):
            learning_rate = 1e-5
            max_grad_norm = None
            optim = tf.compat.v1.train.AdamOptimizer(learning_rate)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                grads_and_vars = optim.compute_gradients(self.loss)
                self.grads = grads_and_vars
                if max_grad_norm is not None:
                    new_grads_and_vars = []
                    for idx, (grad, var) in enumerate(grads_and_vars):
                        if grad is not None:
                            new_grads_and_vars.append((
                                tf.clip_by_norm(grad, max_grad_norm), var))
                    grads_and_vars = new_grads_and_vars
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grad = tf.debugging.check_numerics(
                            grad, "Numerical error in gradient for {}"
                            "".format(var.name))
                    new_grads_and_vars.append((grad, var))
                self.optim = optim.apply_gradients(new_grads_and_vars, 
                                                   global_step=self.global_step)
            # Summarize all gradients
            for grad, var in grads_and_vars:
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradient', grad)

    #######################################################
    def _build_writer(self):
        self.res_dir_te = os.path.join(self.config["res_dir"], self.config["log_dir"])
        self.summary_te = tf.compat.v1.summary.FileWriter(
            os.path.join(self.res_dir_te, "test", "logs"))
        self.saver_best = tf.compat.v1.train.Saver()
    
    #######################################################
    def test(self, data):
        self.checkpoint = root_dir + 'data/sun3d/test/model-cur_best'
        if not os.path.exists(self.checkpoint + ".index"):
            print("Model File {} does not exist! Quitting".format(self.checkpoint))
            exit(1)

        self.saver_best.restore(self.sess, self.checkpoint)
        print("Restoring from the latest checkpoint...")

        cur_global_step = 0 
        test_process('test', self.sess, data, cur_global_step, self.summary_te, self.config,
                     self.x_in_1, self.x_in_2, self.R_in, self.t_in, self.f_in, self.is_training,
                     self.R_hat, self.t_hat, self.logits, self.rec_loss, self.weights,
                     self.config["reg_flag"], self.config["reg_function"])