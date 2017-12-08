# coding: utf-8
import numpy as np
from numpy import array
from os import listdir
from os.path import isfile, join
import re
from random import randint
import datetime
from itertools import islice
import tensorflow as tf
import subprocess
import os.path
import sys

reload(sys)
sys.setdefaultencoding('utf8')

# Config
numDimensions = 300  # Dimensions for each word vector
maxSeqLength = 50  # Max sequence lenght default
batchSize = 24  # Number review for one train
lstmUnits = 64
numClasses = 4
iterationTrain = 10001  # Loop train
iterationTest = 10  # Loop test
numLinePos, numLinePosNeg, numLinePosNegNeu, numLinePosNegNeuFix = 0, 0, 0, 0


# Clean file input
def clean_stop_words(review, stopwords):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords)

    # 3. Remove stop words
    print "Length before: %d" % len(words)
    meaningful_words = [w for w in words if not w in stops]
    print "Length after: %d" % len(meaningful_words)
    #
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


def get_stop_word():
    # Stop words
    stopwords = []
    with open("vn_stopword.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break

            line = line[0].lower()
            stopwords.append(line.replace(" ", "_").replace("\n", ""))
    return stopwords


def clean_text(text):
    # Remove non-letters
    regular_ex = "[^_a-zaăâbcdđeêghiklmnoôơpqrstuưvxyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]"
    emojis = ["😘", "😀", "😁", "😂", "❤", "😃", "😄", "😅", "😆", "😇", "😈", "😉", "😊", "😋", "😌", "😍", "😎", "😏",
              "😐", "😑", "😒", "😓", "😔", "😕", "😖", "😗", "😙", "😚", "😛", "😜", "😝", "😞", "😟​", "😠", "😡",
              "😢", "😣", "​😤", "😥​", "😦", "😧", "😨​", "😩", "😪", "😫​", "😬", "😭", "😮", "😯", "😰", "😱", "😲",
              "😳​", "😴​", "😵", "😶", "😷", "😸​", "😹​", "😺", "😻", "😼", "😽", "😾", "😿", "🙀", "​🙁", "​🙂",
              "​🙃", "​🙄", "​🙅", "​🙆", "​🙇", "​🙈​", "🙉", "​🙊", "​🙋", "​🙌", "​🙍", "​🙎", "​🙏", "❗ ️", "👨",
              "“", "”", "👍", "–", "✅", "…", "💝", "’", "🎈", "❤", "️", "🍎", " ‬"]
    for emoji in emojis:
        text = text.replace(emoji, " ")
    letters = re.sub(regular_ex, " ", text.lower().strip())
    letters = re.sub("[ ]+", " ", letters)
    stop_words = get_stop_word()
    cleaned_stop_word = clean_stop_words(letters, stop_words)
    return cleaned_stop_word


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


if not os.path.exists("glove/inputFile"):
    inputText = ""
    with open("input/inputFile.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            line = clean_text(line[0])
            inputText = inputText + unicode(line, errors='replace') + " "

    with open("glove/inputFile", "w") as f:
        f.write(inputText)
    print("Clean file input done")

# Run script to create wordList and wordVectors
if not os.path.exists("glove/wordList.txt") or not os.path.exists("glove/wordVectors.txt"):
    shell_script = subprocess.Popen(["glove/createWordListAndVector.sh"], stdin=subprocess.PIPE)
    shell_script.stdin.write("yes\n")
    shell_script.stdin.close()
    return_code = shell_script.wait()  # wait to done script

# Load wordList and wordVectors
wordsList = []
with open("glove/wordList.txt", "r") as f:
    while True:
        line = list(islice(f, 1))
        if not line:
            break
        line = line[0].split()
        wordsList.append(line[0])
wordsList = [unicode(word, errors='replace') for word in wordsList]  # Encode words as UTF-8
numWordInput = len(wordsList)
print('Loaded the word list!')
print(numWordInput)

wordVectors = []
with open("glove/wordVectors.txt", "r") as f:
    while True:
        line = list(islice(f, 1))
        if not line:
            break
        items = array(line[0].split()[1:], np.float32)
        wordVectors.append(items)
wordVectors = array(wordVectors[:-1], np.float32)
print ('Loaded the word vectors!')
print(wordVectors.shape)

with tf.Session() as sess:
    pass


# Create idsMatrix
def add_ids(path_file, ids):
    counter = 0
    with open(path_file, "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break

            index_counter = 0
            cleaned_line = unicode(clean_text(line[0]), errors='replace')
            split = cleaned_line.split()
            for word in split:
                try:
                    ids[fileCounter][index_counter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][index_counter] = numWordInput - 1  # Vector for unkown words
                index_counter = index_counter + 1
                if index_counter >= maxSeqLength:
                    break
            counter = counter + 1
    return counter


if not os.path.exists("training/idsMatrix.npy"):
    numWords = []

    with open("input/positive.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            line = unicode(clean_text(line[0]), errors='replace')
            counter = len(line.split())
            numWords.append(counter)
    print('Positive file finished')

    with open("input/negative.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            line = unicode(clean_text(line[0]), errors='replace')
            counter = len(line.split())
            numWords.append(counter)
    print('Negative file finished')

    with open("input/neutral.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            line = unicode(clean_text(line[0]), errors='replace')
            counter = len(line.split())
            numWords.append(counter)
    print('Neutral file finished')

    with open("input/fix.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            line = unicode(clean_text(line[0]), errors='replace')
            counter = len(line.split())
            numWords.append(counter)
    print('Fix file finished')

    numLine = len(numWords)
    averageNumber = sum(numWords) / len(numWords)
    print('The total number of line is', numLine)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', averageNumber)

    maxSeqLength = (averageNumber / 50 + 1) * 50
    with open("temp.txt", "w") as f:
        f.write("Max sequence length:" + str(maxSeqLength))

    ids = np.zeros((numLine, maxSeqLength), dtype='int32')
    fileCounter = 0

    fileCounter += add_ids("input/positive.txt", ids)
    numLinePos = fileCounter

    fileCounter += add_ids("input/negative.txt", ids)
    numLinePosNeg = fileCounter

    fileCounter += add_ids("input/neutral.txt", ids)
    numLinePosNegNeu = fileCounter

    fileCounter += add_ids("input/fix.txt", ids)
    numLinePosNegNeuFix = fileCounter

    # Pass into embedding function and see if it evaluates.
    np.save('training/idsMatrix', ids)
    with open("temp.txt", "a") as f:
        f.write("\nNumber line positive:" + str(numLinePos) + "\nNumber line pos+neg:" + str(
            numLinePosNeg) + "\nNumber line pos+neg+neu:" + str(
            numLinePosNegNeu) + "\nNumber line pos+neg+neu+fix:" + str(numLinePosNegNeuFix))
    print("Done")
else:
    ids = np.load('training/idsMatrix.npy')


def get_train_batch():
    arr_temp = []
    with open("temp.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            arr_temp.append(line[0].split(":")[1])
    maxSeqLength = int(arr_temp[0])
    numLinePos = int(arr_temp[1])
    numLinePosNeg = int(arr_temp[2])
    numLinePosNegNeu = int(arr_temp[3])
    numLinePosNegNeuFix = int(arr_temp[4])
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if i % 4 == 0:
            num = randint(1, numLinePos)
            labels.append([1, 0, 0, 0])
        elif i % 4 == 1:
            num = randint(numLinePos, numLinePosNeg)
            labels.append([0, 1, 0, 0])
        elif i % 4 == 2:
            num = randint(numLinePosNeg, numLinePosNegNeu)
            labels.append([0, 0, 1, 0])
        else:
            num = randint(numLinePosNegNeu, numLinePosNegNeuFix)
            labels.append([0, 0, 0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_test_batch():
    arr_temp = []
    with open("temp.txt", "r") as f:
        while True:
            line = list(islice(f, 1))
            if not line:
                break
            arr_temp.append(line[0].split(":")[1])
    labels = []
    maxSeqLength = int(arr_temp[0])
    numLinePos = int(arr_temp[1])
    numLinePosNeg = int(arr_temp[2])
    numLinePosNegNeu = int(arr_temp[3])
    numLinePosNegNeuFix = int(arr_temp[4])
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(1, numLinePosNegNeuFix)
        if num <= numLinePos:
            labels.append([1, 0, 0, 0])
        elif numLinePosNeg >= num >= numLinePos:
            labels.append([0, 1, 0, 0])
        elif numLinePosNegNeu >= num >= numLinePosNeg:
            labels.append([0, 0, 1, 0])
        else:
            labels.append([0, 0, 0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


#  Init LSTM
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Training to create model
if not os.path.exists("training/models/pretrained_lstm.ckpt-" + str(iterationTrain - 1) + ".index"):
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "training/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterationTrain):
        print("Lan %d" % i)
        # Next Batch of reviews
        nextBatch, nextBatchLabels = get_train_batch()
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if i % 50 == 0:
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if i % ((iterationTrain - 1) / 10) == 0 and i != 0:
            save_path = saver.save(sess, "training/models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()
else:
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('training/models'))

for i in range(iterationTest):
    nextBatch, nextBatchLabels = get_test_batch()
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
