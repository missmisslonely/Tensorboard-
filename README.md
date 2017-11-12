 Tensorboard：可视化学习
Tensorflow做的一些计算是复杂和混乱，就像训练深度神经网络一样。为了更易于理解、调试和优化Tensorflow程序，我们发布了一套可视化工具称为Tensorboard。你可以使用TensorBoard形象化Tensorflow图，绘制图像生成的定量指标图以及附加数据。当TensorBoard完全配置后，是这样的：




本教程的目的是让你掌握TensorBoard的简单用法。当然还有更详细的资料， TensorBoard's GitHub，包括TensorBoard详细用法，技巧和调试信息。

数据序列化
TensorBoard是通过阅读Tensorflow事件文件来操作.Tensorflow事件包含Tensorflow运行所生成的汇总数据。下面是大概的生命周期内TensorBoard汇总数据。

首先，创建你想从中收集汇总数据的TensorFlow图，并决定你想添加哪一个汇总操作。
例如，假设你正在做识别MNIST数字卷积神经网络的训练。你希望记录学习速率是如何随时间变化的，以及目标函数是如何变化的。（通过@{tf.summary.scalar}这个操作分别输出的节点学习速率和损耗，并且将这些输出节点将它们集合起来。）然后，给每一个有意义的 scalar_summary  标签，像 learning rate' or 'loss function’。

也许你也希望可视化激活特定层的分布，或梯度或权重的分布。执行@{tf.summary.histogram}操作就可以收集输出数据，包括梯度输出和保持权重不变的变量。

有关所有可用的摘要操作的详细信息，请查阅文档summary operations.
在Tensorflow上的一些操作不会做什么事情直到你运行它们，或一个运算取决于这些操作的输出。我们刚刚创建的汇总节点围绕着你的图相：任何一个操作都不依赖于它们。因此，为了生成汇总数据，我们需要运行所有这些汇总节点。手工管理他们是乏味的，所以用 @{tf.summary.merge_all} 来将它们组合成一个简单的运算，生成汇总数据。

然后，你就可以运行合并汇总命令，它会依据特点步骤将所有数据生成一个序列化的汇总protobuf对象。最后，将汇总数据写到磁盘，通过汇总protobuf对象传递给@{tf.summary.FileWriter}。

FileWriter 的构造函数包含logdir，这个logdir目录是相当重要的，所有的事件都会写到它所指的目录下。同时， FileWriter 可以采随意生成图 在其构造函数中。如果它接收到一个 graphobject，然后Tensorboard将根据张量形状信息显示你的图像。这会给你一个更好的感受关于图形的生产过程：查看 Tensor shape information

现在已经修改了你的图并且有一个 FileWriter，准备开始你的神经网络吧！如果您愿意，您可以单步运行合并汇总操作，并记录大量的训练数据。不过，这可能比你需要的数据更多。你可以每一百步执行一次汇总。

下面的代码示例是对simple MNIST tutorial的更改,其中我们增加了一些总结，并每十步运行一次。如果你运行这个然后启动tensorboard —logdir=/tmp/tensorflow/mnist,您将能够可视化统计数据，例如在训练过程中权重或精确度是如何变化的。下面的代码是一个摘录，完整的资料在这里。



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

# Do not apply softmax activation yet, see below.
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  # The raw formulation of cross-entropy,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                               reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the
  # raw outputs of the nn_layer above, and then average across
  # the batch.
  diff = tf.nn.softmax_cross_entropy_with_logits(targets=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()

在我们的filewriters初始化后，我们将总结的filewriters作为我们训练和测试模型




# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
你现在就可以使用tensorboard可视化数据了。

启动TensorBoard

运行下面命令，运行TensorBoard。

二者选一
tensorboard —logdir=path/to/log-directory
（python -m tensorboard.main）


logdir 就是 FileWriter 序列化数据的目录。如果这 logdir 包含一个序列化的数据单独运行的子目录，那么Tensorboard将一起展示这些可视化数据。一旦TensorBoard运行，你可以通过你的的Web浏览器 localhost：6006， 查看Tensorboard。
当看着Tensorboard，你会在右上角看到导航标签。每个选项代表一组可以可视化的序列化数据

详细信息关于如何使用 “graph”选项来显示你的图，TensorBoard: Graph Visualization.

更多关于TensorBoard的信息，查看TensorBoard's GitHub.
