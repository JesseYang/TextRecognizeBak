import tensorflow as tf

def hard_loss(loss, hard_sample_num, name):
    '''
    Find the greater loss from the given loss, and return the average loss
    Parameters:
        loss: the loss tensor, should be one dimension (batch_size)
        hard_sample_num: the number of hard samples in each batch
    Return:
        the average loss of the losses selected
    '''

    # loss = tf.squeeze(loss)
    dtype = loss.dtype

    hard_sample_num = tf.minimum(hard_sample_num, tf.shape(loss)[0])
    val, idxes = tf.nn.top_k(loss, k=hard_sample_num)

    min_hard_loss = val[-1]

    hard_mask = loss >= min_hard_loss
    hard_mask = tf.cast(hard_mask, dtype)

    tot_hard_loss = tf.reduce_sum(loss * hard_mask)
    ave_hard_loss = tf.truediv(tot_hard_loss, tf.cast(hard_sample_num, dtype))
    if name != None:
        ave_hard_loss = tf.identity(ave_hard_loss, name=name)

    return ave_hard_loss
