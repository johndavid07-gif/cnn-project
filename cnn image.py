import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Creates a basic Convolutional Neural Network (CNN) for image classification.
    Designed for CIFAR-10 like data.
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    # 1. Load and preprocess data (using CIFAR-10 as an example)
    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # 2. Build the model
    model = create_cnn_model(input_shape=(32, 32, 3), num_classes=10)

    # 3. Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. Train the model
    print("\nStarting training...")
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    # 5. Evaluate the model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # 6. Plot results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
