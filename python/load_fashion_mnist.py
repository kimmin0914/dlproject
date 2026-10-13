# TensorFlow 및 tf.keras 임포트
import tensorflow as tf

# 헬퍼 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt

# 패션 MNIST 데이터셋 임포트 및 로드
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 1. 학습용 및 테스트용 데이터 크기 확인
print("학습용 이미지 데이터 크기:", train_images.shape)
print("학습용 정답(라벨) 개수:", len(train_labels))
print("테스트용 이미지 데이터 크기:", test_images.shape)
print("테스트용 정답(라벨) 개수:", len(test_labels))

# 2. 클래스 이름 정의 (0~9 숫자에 해당하는 실제 옷 이름)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. 픽셀 값을 0 ~ 1 사이로 정규화 (전처리)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 4. 인공신경망 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 5. 모델 컴파일 (학습 환경 설정)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 6. 모델 훈련 (학습)
print("\n[모델 훈련 시작]")
model.fit(train_images, train_labels, epochs=10)

# 7. 모델 평가 및 예측
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

predictions = model.predict(test_images)

# --- [추가된 부분] 시각화를 위한 보조 함수 정의 ---
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue' # 정답을 맞히면 파란색
    else:
        color = 'red'  # 틀리면 빨간색

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
# ---------------------------------------------------

# --- [추가된 부분] 이미지에 제시된 다중 예측 결과 시각화 코드 ---
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    # 이미지 출력 부분
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    
    # 예측 확률 막대그래프 출력 부분
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()
# ----------------------------------------------------------------