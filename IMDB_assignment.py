import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 초기 난수 설정
keras.utils.set_random_seed(42)

# 1. IMDB 데이터 불러오기
(train_data, train_labels), (test_data, test_labels) = (
    keras.datasets.imdb.load_data(num_words=10000)
)

print("훈련 데이터:", len(train_data))
print("테스트 데이터:", len(test_data))

# 2. 입력 길이를 256으로 통일
# 짧은 리뷰는 뒤쪽에 0을 추가하고, 긴 리뷰는 앞부분을 자름
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=0,
    padding="post",
    truncating="pre",
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=0,
    padding="post",
    truncating="pre",
    maxlen=256
)

# 3. 정답 라벨을 원핫 벡터로 변환
# 부정: 0 -> [1, 0]
# 긍정: 1 -> [0, 1]
train_labels = keras.utils.to_categorical(
    train_labels,
    num_classes=2
).astype(np.float32)

test_labels = keras.utils.to_categorical(
    test_labels,
    num_classes=2
).astype(np.float32)

print("훈련 입력 배열 크기:", train_data.shape)
print("테스트 입력 배열 크기:", test_data.shape)
print("훈련 정답 배열 크기:", train_labels.shape)
print("테스트 정답 배열 크기:", test_labels.shape)
print("원핫 정답 예시:")
print(train_labels[:5])

# 각 정답 벡터의 합 확인
assert np.all(train_labels.sum(axis=1) == 1)
assert np.all(test_labels.sum(axis=1) == 1)

# 4. 모델 구성
# 출력 순서: [부정 확률, 긍정 확률]
model = keras.Sequential([
    keras.Input(shape=(256,), dtype="int32"),

    keras.layers.Embedding(
        input_dim=10000,
        output_dim=16,
        mask_zero=True
    ),

    keras.layers.GlobalAveragePooling1D(),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        2,
        activation="softmax"
    )
])

# 원핫 정답에 맞는 손실 함수 사용
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy")
    ]
)

model.summary()

# 5. 훈련 데이터와 검증 데이터 분리
# 훈련: 15,000개 / 검증: 10,000개
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 6. 모델 학습
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# 7. 테스트 데이터 평가
test_loss, test_accuracy = model.evaluate(
    test_data,
    test_labels,
    verbose=2
)

print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.2%}")

# 8. 학습 기록
history_dict = history.history
epochs = range(1, len(history_dict["loss"]) + 1)

# 9. Loss 그래프
plt.figure(figsize=(7, 4))

plt.plot(
    epochs,
    history_dict["loss"],
    "bo",
    label="Training loss"
)

plt.plot(
    epochs,
    history_dict["val_loss"],
    "b-",
    label="Validation loss"
)

plt.title("Training and validation loss")
plt.xlabel("Epoch")
plt.ylabel("Categorical crossentropy")
plt.legend()
plt.tight_layout()
plt.savefig("loss_by_epoch.png", dpi=150)

# 10. Accuracy 그래프
plt.figure(figsize=(7, 4))

plt.plot(
    epochs,
    history_dict["accuracy"],
    "bo",
    label="Training accuracy"
)

plt.plot(
    epochs,
    history_dict["val_accuracy"],
    "b-",
    label="Validation accuracy"
)

plt.title("Training and validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_by_epoch.png", dpi=150)

plt.show()