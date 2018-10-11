import tensorflow as tf

def model_softmax(x,batch_size,total_speakers):
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,1])
    with tf.variable_scope('conv2D_A'):
        x=tf.layers.conv2d(x,filters=32,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_32_A')
        print('Conv2D_A O/P shape ',x.get_shape())
    with tf.variable_scope('conv2D_B'):
        x=tf.layers.conv2d(x,filters=64,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_64')
        print('Conv2D_B O/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,1024])
    x=tf.transpose(x,[1,0,2])
    with tf.variable_scope('LSTM_A'):
        lstm_cell_A = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_A, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_B'):
        lstm_cell_B = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_B, x, dtype=tf.float32)
    print('LSTM O/P shape ',x.get_shape())
    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
    print('Temporal AVG O/P shape ',x.get_shape())
    with tf.variable_scope('Affine_Layer_A'):
        x=tf.layers.dense(x,units=256,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Affine O/P shape ',x.get_shape())
    with tf.variable_scope('Softmax_Layer'):
        x=tf.layers.dense(x,units=total_speakers + 1,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Softmax O/P shape ',x.get_shape())
    return x
def model_softmax_eval(x,batch_size):
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,1])
    with tf.variable_scope('conv2D_A'):
        x=tf.layers.conv2d(x,filters=32,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_32_A')
        print('Conv2D_A O/P shape ',x.get_shape())
    with tf.variable_scope('conv2D_B'):
        x=tf.layers.conv2d(x,filters=64,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_64')
        print('Conv2D_B O/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,1024])
    x=tf.transpose(x,[1,0,2])
    with tf.variable_scope('LSTM_A'):
        lstm_cell_A = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_A, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_B'):
        lstm_cell_B = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_B, x, dtype=tf.float32)
    print('LSTM O/P shape ',x.get_shape())
    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
    print('Temporal AVG O/P shape ',x.get_shape())
    with tf.variable_scope('Affine_Layer_A'):
        x=tf.layers.dense(x,units=256,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Affine O/P shape ',x.get_shape())
    #with tf.variable_scope('Softmax_Layer'):
    #    x=tf.layers.dense(x,units=total_speakers + 1,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #print('Softmax O/P shape ',x.get_shape())
    return x


def model(x,batch_size,total_speakers):
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,1])
    with tf.variable_scope('conv2D'):
        x=tf.layers.conv2d(x,filters=32,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_16')
        print('Conv2D O/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,1024])
    x=tf.transpose(x,[1,0,2])
    with tf.variable_scope('LSTM_A'):
        lstm_cell_A = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_A, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_B'):
        lstm_cell_B = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_B, x, dtype=tf.float32)
    print('LSTM O/P shape ',x.get_shape())
    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
    print('Temporal AVG O/P shape ',x.get_shape())
    with tf.variable_scope('Affine_Layer_A'):
        x=tf.layers.dense(x,units=256,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Affine O/P shape ',x.get_shape())
    #with tf.variable_scope('Softmax_Layer'):
    #    x=tf.layers.dense(x,units=total_speakers + 1,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #print('Softmax O/P shape ',x.get_shape())
    return x

def model_eval(x,batch_size):
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,1])
    with tf.variable_scope('conv2D'):
        x=tf.layers.conv2d(x,filters=32,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_16')
        print('Conv2D O/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,1024])
    x=tf.transpose(x,[1,0,2])
    with tf.variable_scope('LSTM_A'):
        lstm_cell_A = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_A, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_B'):
        lstm_cell_B = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_B, x, dtype=tf.float32)
    print('LSTM O/P shape ',x.get_shape())
    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
    print('Temporal AVG O/P shape ',x.get_shape())
    with tf.variable_scope('Affine_Layer_A'):
        x=tf.layers.dense(x,units=256,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Affine O/P shape ',x.get_shape())
    #with tf.variable_scope('Softmax_Layer'):
    #    x=tf.layers.dense(x,units=total_speakers + 1,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #print('Softmax O/P shape ',x.get_shape())
    return x

def model_b1(x,batch_size,total_speakers):
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,1])
    with tf.variable_scope('conv2D'):
        x=tf.layers.conv2d(x,filters=64,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_16')
        print('Conv2D O/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,2048])
    x=tf.transpose(x,[1,0,2])
    with tf.variable_scope('LSTM_A'):
        lstm_cell_A = tf.nn.rnn_cell.BasicLSTMCell(1024,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_A, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_B'):
        lstm_cell_B = tf.nn.rnn_cell.BasicLSTMCell(1024,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_B, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_C'):
        lstm_cell_C = tf.nn.rnn_cell.BasicLSTMCell(1024,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_C, x, dtype=tf.float32)
    with tf.variable_scope('LSTM_D'):
        lstm_cell_D = tf.nn.rnn_cell.BasicLSTMCell(1024,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell_D, x, dtype=tf.float32)
    print('LSTM O/P shape ',x.get_shape())
    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
    print('Temporal AVG O/P shape ',x.get_shape())
    with tf.variable_scope('Affine_Layer'):
        x=tf.layers.dense(x,units=512,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Affine O/P shape ',x.get_shape())
    with tf.variable_scope('Softmax_Layer'):
        x=tf.layers.dense(x,units=total_speakers + 1,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Softmax O/P shape ',x.get_shape())
    return x

def model_b2(x,batch_size,total_speakers):
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,1])
    with tf.variable_scope('conv2D'):
        x=tf.layers.conv2d(x,filters=16,kernel_size=5, strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),activation=tf.nn.relu,name='conv2D_16')
        print('Conv2D O/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,512])
    x=tf.transpose(x,[1,0,2])
    with tf.variable_scope('LSTM'):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        x, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    print('LSTM O/P shape ',x.get_shape())
    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
    print('Temporal AVG O/P shape ',x.get_shape())
    with tf.variable_scope('Affine_Layer'):
        x=tf.layers.dense(x,units=128,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Affine O/P shape ',x.get_shape())
    with tf.variable_scope('Softmax_Layer'):
        x=tf.layers.dense(x,units=total_speakers + 1,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print('Softmax O/P shape ',x.get_shape())
    return x

