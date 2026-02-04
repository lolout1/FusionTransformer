import tensorflow as tf

def keras_to_h5(keras_path, h5_path):
    # Load the .keras model
    model = tf.keras.models.load_model(keras_path)

    # Save the model in HDF5 (.h5) format
    model.save(h5_path, save_format="h5")

    print(f"Model converted: {keras_path} → {h5_path}")


# Example usage
if __name__ == "__main__":
    keras_model_path = "model_for_training.keras"
    h5_model_path = "best_model_test.h5"

    keras_to_h5(keras_model_path, h5_model_path)
