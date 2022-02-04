import tflite_runtime.interpreter as tflite
import numpy as np
import time


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def setup_inference(args):

    # Load the modelfile
    interpreter = tflite.Interpreter(model_path=args["model_file"], num_threads=args["num_threads"])

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    return (width, height, interpreter)


def extract_label(best_result_idx, labels):

    split_string = labels[best_result_idx].split(",", 1)[0]

    label = split_string.split(":", 2)[1]

    return label


def convert_to_float32(data, num_channels, height, width):

    data = np.frombuffer(data, dtype=np.uint8)

    data = np.reshape(
        data, newshape=(1, height, width, num_channels))

    data = (np.float32(data) * (1.0 / 127.5)) - 1.0

    data = data.view('<f4')

    return data

def convert_to_uint8(data, num_channels, height, width):

    data = np.frombuffer(data, dtype=np.uint8)

    data = np.reshape(
        data, newshape=(1, height, width, num_channels))

    # data = np.expand_dims(data, axis=0)

    # data = np.uint8((np.uint8(data) * (1.0 / 127.5)) - 1.0)

    return data    


def get_bounding_boxes(interpreter, input_data, height, width, labels, num_channels, bbx_dict):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    uint8_data = convert_to_uint8(input_data, num_channels, height, width)

    interpreter.set_tensor(input_details[0]['index'], uint8_data)

    start_time = time.time()
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * 720)))
            xmin = int(max(1,(boxes[i][1] * 1280)))
            ymax = int(min(height,(boxes[i][2] * 720)))
            xmax = int(min(width,(boxes[i][3] * 1280)))

            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

            bbx_dict[str(object_name)] = (xmin, xmax, ymin, ymax)

    return bbx_dict
