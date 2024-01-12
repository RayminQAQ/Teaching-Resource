import tensorflow as tf

# 資料預處理: 引入mnist資料集
(x_train_image, y_train_label), (x_test_image, y_test_label) = tf.keras.datasets.mnist.load_data()

# Build model
# CNN: mnist shape(28x28x1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# FNN: fully connect network
model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(28*28, activation='relu'))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='relu'))

# Select compile method: optimizer, loss function
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train & Test the model
history = model.fit(x_train_image, y_train_label, epochs=10,
                    validation_data=(x_test_image, y_test_label))

# Model Structure
print("Model Structure")
model.summary()

# Model train && validation history
print("Model Performance Analysis")
print(history.history)
print(f"The Overall Accuracy: {history.history['loss']}")
print(f"The Overall validation Accuracy: {history.history['val_accuracy']}")

"""
Useful tutorial:
- https://hackmd.io/@zengyu/r1jBeLRQh
- https://blog.csdn.net/weixin_46072771/article/details/108591263
- https://keras.io/zh/models/model/
"""