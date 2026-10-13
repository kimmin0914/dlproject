import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 1000

# 1. 데이터 로드 및 전처리
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

train_labels = np.expand_dims(train_labels, axis=-1)
test_labels = np.expand_dims(test_labels, axis=-1)

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# 첫 번째 데이터 확인용 창 (창을 닫아야 학습이 시작됩니다)
plt.plot(train_data[0])
plt.show()


# 2. 모델 구성 및 컴파일
# (1) 기준 모델
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

# (2) 작은 모델
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

# (3) L2 정규화 모델
l2_model = keras.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

# (4) 드롭아웃 모델
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
dpt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

# (5) L2 + 드롭아웃 결합(섞은) 모델
l2_dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
l2_dpt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])


# 3. 모델 학습 (모두 epochs=25로 고정)
baseline_history = baseline_model.fit(train_data, train_labels, epochs=25, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
smaller_history = smaller_model.fit(train_data, train_labels, epochs=25, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
l2_history = l2_model.fit(train_data, train_labels, epochs=25, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
dpt_model_history = dpt_model.fit(train_data, train_labels, epochs=25, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
l2_dpt_history = l2_dpt_model.fit(train_data, train_labels, epochs=25, batch_size=512, validation_data=(test_data, test_labels), verbose=2)


# 4. 결과 시각화 함수 (소문자 표기 반영)
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name + ' val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name + ' train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()

# 비교할 모델들을 하나의 리스트로 묶기 (bigger 제외)
models_to_compare = [
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('l2', l2_history),
    ('dropout', dpt_model_history),
    ('l2+dropout', l2_dpt_history)
]

# 첫 번째 그래프: 손실(Loss - binary_crossentropy) 비교
plot_history(models_to_compare, key='binary_crossentropy')

# 두 번째 그래프: 정확도(Accuracy) 비교
plot_history(models_to_compare, key='accuracy')