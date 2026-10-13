import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# --- [수정 1] 정답지(레이블)를 원-핫 벡터로 변환 ---
# 0은 [1. 0.]으로, 1은 [0. 1.] 형태의 벡터로 바뀝니다.
print("훈련 레이블: {}".format(train_labels[0]))

# train_labels = keras.utils.to_categorical(train_labels)
# test_labels = keras.utils.to_categorical(test_labels)

new_train_labels = []
for label in train_labels:
    if label == 0:
        new_train_labels.append([1., 0.])
    else:
        new_train_labels.append([0., 1.])
train_labels = np.array(new_train_labels)

new_test_labels = []
for label in test_labels:
    if label == 0:
        new_test_labels.append([1., 0.])
    else:
        new_test_labels.append([0., 1.])
test_labels = np.array(new_test_labels)
# ---------------------------------------------------

print("훈련 레이블: {}".format(train_labels[0]))

print("훈련 데이터: {}, 훈련 레이블: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
# 주의: 질문하신 코드에 특수 기호가 지워져 있어서 다시 복원했습니다.
word_index[""] = 0
word_index[""] = 1
word_index[""] = 2
word_index[""] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

# 빈칸 기호 복원
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index[""], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index[""], padding="post", maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# --------------------------------------------------------

x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()