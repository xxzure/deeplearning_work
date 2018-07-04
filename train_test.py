import pandas as pd
import tensorflow as tf
import os

from models.textCNN import TextCNN
from models.textRNN import TextRNN
from models.textRCNN import TextRCNN
from data import data_process
import argparse

# hyper parameter
def get_args():
    parser = argparse.ArgumentParser(
        description="model parameter initial value")
    parser.add_argument('--model', default="TextCNN", type=str,
                        help='default train model name')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='steps to train model over')
    parser.add_argument('--num_epoches', default=30, type=int,
                        help='steps to train model over')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output_dir', default="output",
                        type=str, help='output submission folder')
    return parser.parse_args()


if __name__ == '__main__':
    # get args parameter
    args = get_args()
    learning_rate = args.lr
    batch_size = args.batch_size
    #143613 total dataset
    num_epoches = args.num_epoches
    output_dir = args.output_dir
    model = args.model

    # initilize the model
    W_list = data_process.get_embedding()
    net = eval(model)(W_list,batch_size=batch_size)

    # set the loss and optimizer
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=net.p, labels=net.Y))
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    auc = tf.metrics.auc(net.Y, tf.nn.sigmoid(net.p))

    # allow gpu growth increase dynamically
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # run variables_initializer
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # training
    print("Loading data")

    # Train data iterator
    features_placeholder = tf.placeholder(data_process.X_tra.dtype, data_process.X_tra.shape)
    labels_placeholder = tf.placeholder(data_process.y_tra.dtype, data_process.y_tra.shape)
    dataset=tf.data.Dataset.from_tensor_slices((features_placeholder,labels_placeholder))
    dataset=dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.shuffle(buffer_size=10000).repeat(count=None)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Valid data iterator
    features_val_placeholder = tf.placeholder(data_process.X_val.dtype, data_process.X_val.shape)
    labels_val_placeholder = tf.placeholder(data_process.y_val.dtype, (data_process.y_val.shape))
    dataset_val = tf.data.Dataset.from_tensor_slices((features_val_placeholder,labels_val_placeholder))
    dataset_val=dataset_val.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset_val = dataset_val.repeat(count=None)
    iterator_val = dataset_val.make_initializable_iterator()
    next_element_val = iterator_val.get_next()

    sess.run(iterator.initializer,feed_dict={features_placeholder:data_process.X_tra,labels_placeholder:data_process.y_tra})
    sess.run(iterator_val.initializer,feed_dict={features_val_placeholder:data_process.X_val,labels_val_placeholder:data_process.y_val})

    # train_batches = data_process.batch_iter("train", batch_size, num_epoches)
    print("Loading data Success")
    num_batches_per_epoch = int((len(data_process.X_tra) - 1) / batch_size)
    num_batches_per_epoch_val = int((len(data_process.X_val) - 1) / batch_size)
    for _ in range(num_epoches*num_batches_per_epoch):
        # x_batch, y_batch = zip(*batch)

        train_x_batch,train_y_batch=sess.run(next_element)
        _, step, batch_loss, auc_value = sess.run([train_op, global_step, loss, auc], feed_dict={net.X: train_x_batch, net.Y: train_y_batch, net.keep: 0.5})
        current_step = tf.train.global_step(sess, global_step)
        if current_step%10==0:
            print("Step",current_step)
        if current_step % num_batches_per_epoch == 0:
            print("epoch:{:0f}, train-loss:{:4f}, auc".format(current_step / num_batches_per_epoch, batch_loss),  auc_value)
            # x_valid, y_valid = data_process.X_val, data_process.y_val
            val_acc=.0
            val_loss=.0
            for _ in range(num_batches_per_epoch_val):
                val_x_batch,val_y_batch=sess.run(next_element_val)
                feed_dict_val={net.X: val_x_batch, net.Y: val_y_batch, net.keep: 1}
                val_batch_loss, val_auc_value=sess.run([loss, auc], feed_dict=feed_dict_val)
                val_loss+=val_batch_loss
                val_acc+=val_auc_value[0]
            val_loss/=num_batches_per_epoch_val
            val_acc/=num_batches_per_epoch_val

            print("val-loss:{:4f}, auc:".format(val_loss),val_loss)

    # log the submission
    submission = pd.read_csv(os.path.join("data", "sample_submission.csv"))
    test_batch_size = batch_size
    test_batches = data_process.batch_iter("test", test_batch_size, 1, shuffle=False)
    start = 0
    for batch in test_batches:
        x_batch = batch
        y_pred = sess.run(tf.nn.sigmoid(net.p), feed_dict={net.X: x_batch, net.keep: 1})
        submission.loc[start:start+test_batch_size-1, ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        start += test_batch_size
    submission.to_csv('sub.csv', index=False)
    sess.close()
