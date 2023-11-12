import tensorflow as tf
from PIL import Image
import argparse

def preprocess_img(image, img_shape=(224, 224)):
    """
    1. Converts image Type from 'uint8' -> 'float32'
    2. Reshapes image to [img_shape, img_shape, color_channels]
    """
    processed_img = tf.cast(image, tf.float32)
    processed_img = tf.image.resize(processed_img, img_shape)
    processed_img = tf.expand_dims(processed_img, axis=0)

    return processed_img

# Function to read a text file and convert its contents to a list
def read_txt_to_list(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    # Remove any newline characters and return the list
    return [line.strip() for line in content]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='recognize_food',
        description='Predicts a food')
    
    parser.add_argument('--image', type=str, required=True,
                        help='image path to predict')

    args = parser.parse_args()
    print(args.image)
    labels_list = read_txt_to_list("labels.txt")
    # Load model
    food_101_model = tf.keras.models.load_model("model")
    my_img = Image.open(args.image)

    pred_prob = food_101_model.predict(preprocess_img(image=my_img))
    pred_class = labels_list[pred_prob.argmax()]
    print(pred_class)

