import tflite_runtime.interpreter as tflite
import numpy as np
import time


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def setup_inference(args):

    # Load the modelfile
    interpreter = tflite.Interpreter(model_path=args["model_file"])

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


def get_labels(interpreter, input_data, height, width, labels):

    num_channels = 3

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    float_data = convert_to_float32(input_data, num_channels, height, width)

    interpreter.set_tensor(input_details[0]['index'], float_data)

    start_time = time.time()
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]

    stop_time = time.time()

    print('time: {:.3f} ms'.format((stop_time - start_time) * 1000))

    best_result = top_k[0]

    label = extract_label(best_result, labels)

    score = float(results[best_result])

    return score, label
