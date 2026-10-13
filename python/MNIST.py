import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Training data load
all_files = []
for i in range(0, 10):
    path_dir = './images/training/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)

x_train_datas = []
y_train_datas = []

for num in range(0, 10):
    for numbers in all_files[num]:
        img_path = './images/training/{0}/{1}'.format(num, numbers)
        print("load: " + img_path)
        img = Image.open(img_path)
        imgarr = np.array(img, dtype=np.float32)
        x_train_datas.append(imgarr.flatten())
        
        y_tmp = np.zeros(shape=(10,), dtype=np.float32)
        y_tmp[num] = 1.0
        y_train_datas.append(y_tmp)

# 2. Testing data load
eval_files = []
for i in range(0, 10):
    path_dir = './images/testing/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    eval_files.append(file_list)

x_test_datas = []
y_test_datas = []

for num in range(0, 10):
    for numbers in eval_files[num]:
        img_path = './images/testing/{0}/{1}'.format(num, numbers)
        print("load: " + img_path)
        img = Image.open(img_path)
        imgarr = np.array(img, dtype=np.float32)
        x_test_datas.append(imgarr.flatten())
        
        y_tmp = np.zeros(shape=(10,), dtype=np.float32)
        y_tmp[num] = 1.0
        y_test_datas.append(y_tmp)

print("Train images:", len(x_train_datas))
print("Train labels:", len(y_train_datas))
print("Test images:", len(x_test_datas))
print("Test labels:", len(y_test_datas))

# 리스트를 NumPy 배열 형태로 변환
x_train_datas = np.array(x_train_datas)
y_train_datas = np.array(y_train_datas)
x_test_datas = np.array(x_test_datas)
y_test_datas = np.array(y_test_datas)

# --- [수정된 부분] 0 ~ 255 범위의 픽셀 값을 -1 ~ 1 범위로 정규화 ---
x_train_datas = (x_train_datas / 127.5) - 1.0
x_test_datas = (x_test_datas / 127.5) - 1.0
# ----------------------------------------------------------------------

# 3. Model build
inputs = tf.keras.Input(shape=(784,), name="Input")
hidden = layers.Dense(512, activation="relu", name="Hidden1")(inputs)
outputs = layers.Dense(10, activation="softmax", name="Output")(hidden)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

# 4. Train & Evaluate
model.fit(x_train_datas, y_train_datas, epochs=5, shuffle=True)

test_loss, test_acc = model.evaluate(x_test_datas, y_test_datas)
print('테스트 정확도:', test_acc)