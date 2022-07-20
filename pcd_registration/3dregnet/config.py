#######################################################
def get_config():
    
    config = {}
    config["data_dir_te"] = ['sun3d']
    config["data_te"] = 'sun3d'
    config["res_dir"] = './logs'                        # Base directory for results
    config["log_dir"] = 'log_lie'                       # Save directory name inside results
    config["report_intv"] = 100                         # Summary interval
    config["gpu_options"] = 'cpu'                       # Choose between gpu, cpu
    config["gpu_number"] = '0'                          # Choose which gpu number
    config["net_depth"] = 12                            # Number of layers
    config["net_nchannel"] = 128                        # Number of channels in a layer
    config["net_act_pos"] = 'post'                      # Position of the activation (choose between 'pre', 'mid', 'post')
    config["net_gcnorm"] = True                         # Context normalization for each layer
    config["net_batchnorm"] = True                      # Batch normalization
    config["net_bn_test_is_training"] = False           # is_training value for testing
    config["net_concat_post"] = False                   # Retrieve top k values or concat from different layers
    config["loss_function"] = 'l1'                      # Choose between l1, l2, wls, gm, l05
    config["loss_classif"] = 0.5                        # Weight of the classification loss
    config["loss_reconstruction"] = 0.01                # Weight of the essential loss
    config["loss_reconstruction_init_iter"] = 20000     # Initial iterations to run only the classification loss
    config["loss_decay"] = 0.0                          # l2 decay
    config["reg_flag"] = False                          # Refine transformation
    config["reg_function"] = 'global'                   # Registration function (choose between 'fast' and 'global')
    config["representation"] = 'lie'                    # Type of Representation (choose between 'lie', 'quat' and 'linear')

    return config
