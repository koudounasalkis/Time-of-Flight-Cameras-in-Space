import tensorflow as tf
from sklearn.model_selection import KFold
import time
import tqdm
import numpy as np
import coarseCNN as cCNN
import fineCNN as fCNN
import prepare_dataset_cnn as dataset

optimizer = tf.keras.optimizers.Adam(learning_rate=5*0.000001)
loss_fn = tf.keras.losses.MeanAbsoluteError()

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)


#######################################################
def train_and_evaluate_model(x, y, x_v, y_v, epochs):

    for i in range(0, epochs):
        with tf.GradientTape() as coarse_tape, tf.GradientTape() as fine_tape:
    
          logits1 = cCNN.coarseCNN(x, training=True)
          logits2 = fCNN.coarsefineCNN(x, training=True)

          loss_coarse = tf.reduce_sum(tf.abs(tf.subtract(logits1, y)))
          loss_fine = tf.reduce_sum(tf.abs(tf.subtract(logits2, y)))
          loss_sup = tf.add(loss_coarse, loss_fine)

        c_gradients = coarse_tape.gradient(loss_sup, cCNN.coarseCNN.trainable_weights)
        f_gradients = fine_tape.gradient(loss_sup, fCNN.coarsefineCNN.trainable_weights)

        optimizer.apply_gradients(zip(c_gradients, cCNN.coarseCNN.trainable_weights))
        optimizer.apply_gradients(zip(f_gradients, fCNN.coarsefineCNN.trainable_weights))
      
        logits1_v = cCNN.coarseCNN(x_v)
        logits2_v = fCNN.coarsefineCNN(x_v)

        loss_coarse_v = tf.reduce_sum(tf.abs(tf.subtract(logits1_v, y_v)))
        loss_fine_v = tf.reduce_sum(tf.abs(tf.subtract(logits2_v, y_v)))
        val_loss = tf.add(loss_coarse_v, loss_fine_v)

        print('\r\t\t\tEpoch: %d || train_loss: %f - val_loss: %f'%(i+1, loss_sup, val_loss), end='')


#######################################################
def train_coarsefineCNN(x, y, epochs):

    for i in range(0, epochs):
        with tf.GradientTape() as coarse_tape, tf.GradientTape() as fine_tape:
    
          train = x
          logits1 = cCNN.coarseCNN(train, training=True)
          logits2 = fCNN.coarsefineCNN(train, training=True)

          loss_coarse = loss_fn(logits1, y)
          loss_fine = loss_fn(logits2, y)
          loss_sup = tf.add(loss_coarse, loss_fine)

        c_gradients = coarse_tape.gradient(loss_sup, cCNN.coarseCNN.trainable_weights)
        f_gradients = fine_tape.gradient(loss_sup, fCNN.coarsefineCNN.trainable_weights)

        optimizer.apply_gradients(zip(c_gradients, cCNN.trainable_weights))
        optimizer.apply_gradients(zip(f_gradients, fCNN.coarsefineCNN.trainable_weights))

        print('\r\t\t\tEpoch: %d || train_loss: %f'%(i+1, loss_fine), end='')

    return loss_sup


#######################################################
def evaluate_coarsefineCNN(x_v, y_v, epochs):

    for i in range(0, epochs):
      
        logits1_v = cCNN.coarseCNN(x_v)
        logits2_v = fCNN.coarsefineCNN(x_v)

        loss_coarse_v = loss_fn(logits1_v, y_v)
        loss_fine_v = loss_fn(logits2_v, y_v)
        val_loss = tf.add(loss_coarse_v, loss_fine_v)

        print('\r\t\t\tEpoch: %d || val_loss: %f'%(i+1, val_loss), end='')

    return val_loss


#######################################################
def main(): 
    checkpointcCNN = tf.train.Checkpoint(cCNN.coarseCNN)
    managerc = tf.train.CheckpointManager(checkpointcCNN, '/content/gdrive/My Drive/ckpts_cCNN', max_to_keep=2)

    checkpointcfCNN = tf.train.Checkpoint(fCNN.coarsefineCNN)
    managercf = tf.train.CheckpointManager(checkpointcfCNN, '/content/gdrive/My Drive/ckpts_cfCNN', max_to_keep=2)

    checkpointcCNN.restore(tf.train.latest_checkpoint('/content/gdrive/My Drive/ckpts_cCNN'))
    checkpointcfCNN.restore(tf.train.latest_checkpoint('/content/gdrive/My Drive/ckpts_cfCNN'))

    fold_no = 1
    epochs = 100
    partial_train_loss = []
    train_losses = []
    partial_val_loss = []
    val_losses = []

    igp, gtp, dataset_size, batch_size = dataset.prepare_dataset()

    for train_index, test_index in kfold.split(igp):

        print('--------------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        batch_no = 1

        for batch in range(dataset_size // batch_size):
        
            print(f'\tBatch {batch_no}:')
        
            for i in tqdm(range(train_index.size), desc='\t\tTraining...'): 
                x = igp[train_index][i]
                y = np.array(gtp)[train_index][i]
                start = time.time()
                train_loss = train_coarsefineCNN(x,y,epochs)
                partial_train_loss.append(train_loss)
                interval = time.time() - start
                print(f'Train Loss over {epochs} epochs: {np.mean(partial_train_loss)}')
            
            for j in tqdm(range(test_index.size), desc='\t\tValidating...'):
                x_v = igp[test_index][j]
                y_v = np.array(gtp)[test_index][j]
                start = time.time()
                val_loss = evaluate_coarsefineCNN(x_v, y_v, epochs)
                partial_val_loss.append(val_loss)
                val_loss = train_and_evaluate_model(x, y, x_v, y_v, 100)
                interval = time.time() - start
                print(f'Validation Loss over {epochs} epochs: {np.mean(partial_val_loss)}')

            # Increase batch number  
            train_losses.append(np.mean(partial_train_loss))
            print(f'\tTrain Loss over batch {batch_no}: {np.mean(partial_train_loss)}')
            val_losses.append(np.mean(partial_val_loss))
            print(f'\tValidation Loss over batch {batch_no}: {np.mean(partial_val_loss)}')
            batch_no += 1

        # Increase fold number
        print(f'Train Loss of fold {fold_no}: {np.mean(train_losses)}')
        print(f'Validation Loss of fold {fold_no}: {np.mean(val_losses)}')
        fold_no += 1
        managerc.save()
        managercf.save()