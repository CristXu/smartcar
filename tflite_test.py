import tensorflow as tf 
import numpy as np 
import cv2 as cv 


def run_keras_model(keras_file, test_images):
    model = tf.keras.models.load_model(keras_file)
    if(len(test_images.shape) == 3):
        test_images = test_images[...,np.newaxis]
    
    results = model.predict(test_images / 128.0 - 1).argmax(axis=-1)
    resutls = results.squeeze() 
    return results

# Helper function to run inference on a TFLite model
labels = ["cat", "dog", "horse", "pig", "casttle"] + ["apple", "orange", "banana", "liulian", "grape"]
def run_tflite_model(tflite_file, test_images):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_images),), dtype=int)

    #Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_images = test_images * input_scale + input_zero_point
    if input_details['dtype'] == np.float32:
        test_images = test_images / 128.0 - 1

    cv.namedWindow("img", 0)
    for i, test_image in enumerate(test_images):
        # cv.imshow("img", cv.resize(test_image, (128,128)))
        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()
        print("Predict %s"%labels[predictions[i]])
        # cv.waitKey(0)

    return np.asarray(predictions).astype("uint8")

if __name__ == "__main__":
    data = np.load(r'./test_x.npy')
    label = np.load(r'./test_y.npy').flatten()

    seed = np.arange(0, len(label))
    np.random.shuffle(seed)
    data = data[seed]
    label = label[seed]

    result_tf = run_tflite_model(r"./models/model_22_0.6979_quant.tflite", data)
    acc_tf = (result_tf == label).sum() / len(label)

    result_keras = run_keras_model(r"./models/model_22_0.6979.h5", data)
    acc_keras = (result_keras == label).sum() / len(label)
    print(acc_tf, acc_keras)