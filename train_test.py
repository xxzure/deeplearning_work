from models import textCNN
from data import data_process
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    batch_size = 256
    num_epoches = 3
    W_list = data_process.get_embedding()
    net = textCNN.network(W_list)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.p, labels=net.Y))
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.0001)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    auc = tf.metrics.auc(net.Y, net.p)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    train_batches = data_process.batch_iter("train", batch_size, num_epoches)
    num_batches_per_epoch = int((len(data_process.X_tra) - 1) / batch_size) + 1
    for batch in train_batches:
        x_batch, y_batch = zip(*batch)
        _, step, batch_loss, auc_value = sess.run([train_op, global_step, loss, auc],feed_dict={net.X: x_batch, net.Y: y_batch, net.keep: 0.5})
        current_step = tf.train.global_step(sess, global_step)
        if current_step % num_batches_per_epoch == 0:
            print("epoch:", current_step / num_batches_per_epoch, "train-loss:", batch_loss, "auc:", auc_value)
            x_valid, y_valid = data_process.X_val, data_process.y_val
            step, val_batch_loss, auc_value = sess.run([global_step, loss, auc],feed_dict={net.X: x_valid, net.Y: y_valid, net.keep:1})
            print("val-loss:", batch_loss, "auc:", auc_value)

    submission = pd.read_csv('sample_submission.csv')
    test_batches = data_process.batch_iter("test", batch_size, 1, shuffle=False)
    for batch in test_batches:
        x_batch = zip(*batch)
        y_pred = sess.run(net.p,feed_dict={net.X: x_batch, net.keep: 1})
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv('submission.csv', index=False)
    sess.close()
