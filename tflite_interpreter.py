import tflite_runtime.interpreter as tflite
import numpy as np
import time

classes_names = ["2_clubs","2_diamonds","2_hearts","2_spades",\
               "3_clubs","3_diamonds","3_hearts","3_spades",\
               "4_clubs","4_diamonds","4_hearts","4_spades",\
               "5_clubs","5_diamonds","5_hearts","5_spades",\
               "6_clubs","6_diamonds","6_hearts","6_spades",\
               "7_clubs","7_diamonds","7_hearts","7_spades",\
               "8_clubs","8_diamonds","8_hearts","8_spades",\
               "9_clubs","9_diamonds","9_hearts","9_spades",\
               "10_clubs","10_diamonds","10_hearts","10_spades",\
               "ace_clubs","ace_diamonds","ace_hearts","ace_spades",\
               "jack_clubs","jack_diamonds","jack_hearts","jack_spades",\
               "king_clubs","king_diamonds","king_hearts","king_spades",\
               "queen_clubs","queen_diamonds","queen_hearts","queen_spades"]

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

    data = data.astype('float32')

    data = np.reshape(
        data, newshape=(height, width, num_channels))

    #stupid way to rotate 270 degs
    data = np.rot90(data)
    data = np.rot90(data)
    data = np.rot90(data)

    data = np.reshape(
        data, newshape=(1, height , width, num_channels))

    return data


def get_labels(interpreter, input_data, height, width, num_channels):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    float_data = convert_to_float32(input_data, num_channels, height, width)

    interpreter.set_tensor(input_details[0]['index'], float_data)

    start_time = time.time()
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    stop_time = time.time()

    print('Inference time: {:.3f} ms'.format((stop_time - start_time) * 1000))

    best_result = results.argmax()

    return results[best_result], classes_names[best_result]
