from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GObject
import argparse
import sys
import traceback
import numpy as np
import tflite_runtime.interpreter as tflite
import platform
from PIL import Image
import time
import gstreamer.utils as utils

video_width = 1280
video_height = 720
framerate = 60

flip_values = {
    "0": 0,
    "90": 1,
    "180": 2,
    "270": 3
}


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def setup_inference(args, interpreter):

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    return (width, height)


def extract_label(best_result_idx, labels):

    split_string = labels[best_result_idx].split(",", 1)[0]

    label = split_string.split(":", 2)[1]

    return label


def convert_to_float32(data, num_channels):

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

    float_data = convert_to_float32(input_data, num_channels)

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


def new_sample_callback(appsink, data):

    interpreter = data[0]

    overlay = data[1]

    labels = data[2]

    sample = appsink.emit("pull-sample")

    buffer = sample.get_buffer()

    caps_format = sample.get_caps().get_structure(0)

    frmt_str = caps_format.get_value('format')
    video_format = GstVideo.VideoFormat.from_string(frmt_str)

    width, height = caps_format.get_value(
        'width'), caps_format.get_value('height')

    ret, map_info = buffer.map(Gst.MapFlags.READ)

    results = get_labels(
        interpreter, map_info.data, width, height, labels)

    overlay.set_property(
        "text", "{} - {:.2f}%".format(results[1], results[0]*100))

    return Gst.FlowReturn.OK


ap = argparse.ArgumentParser()

ap.add_argument("-ip", "--ip_addr", required=True,
                default="127.0.0.1", help="IP to stream to")

ap.add_argument("-flip", "--flip", required=False,
                default="0", help="video flip in degrees")

ap.add_argument(
    '-m',
    '--model_file',
    default='mobilenet_v1_1.0_224_quantized_1_default_1.tflite',
    help='.tflite model to be executed',
    required=True)

ap.add_argument(
    '--num_threads', default=2, type=int, help='number of threads')

ap.add_argument(
    '-l',
    '--label_file',
    help='name of file containing labels',
    required=True)


args = vars(ap.parse_args())

#Load the modelfile
interpreter = tflite.Interpreter(model_path=args["model_file"])

#Sets up the inference enginer
width, height = setup_inference(args, interpreter)

#Init Gstreamer plugin
Gst.init(None)


#Create all gst elements and set the necessary properties

src = Gst.ElementFactory.make("nvarguscamerasrc")
src.set_property("sensor_id", 0)

camera_caps_str = "video/x-raw(memory:NVMM),width={}, height={}, framerate={}/1, format=NV12".format(
    video_width, video_height, framerate)

camera_caps = Gst.Caps.from_string(camera_caps_str)
camera_filter = Gst.ElementFactory.make("capsfilter", "filter")
camera_filter.set_property("caps", camera_caps)

nv_vid_conv = Gst.ElementFactory.make("nvvidconv")
nv_vid_conv.set_property("flip-method", flip_values[args["flip"]])

h_264_enc = Gst.ElementFactory.make("omxh264enc")

appsink = Gst.ElementFactory.make("appsink", "sink")
appsink.set_property("emit-signals", True)

videoconvert_1 = Gst.ElementFactory.make("videoconvert")
videoconvert_2 = Gst.ElementFactory.make("nvvidconv")

tee = Gst.ElementFactory.make("tee")

rtp_264_pay = Gst.ElementFactory.make("rtph264pay")
rtp_264_pay.set_property("config-interval", 1)
rtp_264_pay.set_property("pt", 96)

udp_sink = Gst.ElementFactory.make("udpsink")
udp_sink.set_property("host", args["ip_addr"])
udp_sink.set_property("port", 5000)

queue_1 = Gst.ElementFactory.make("queue")
queue_2 = Gst.ElementFactory.make("queue")

queue_1.set_property("max-size-buffers", 1)
queue_1.set_property("leaky", 2)

queue_2.set_property("max-size-buffers", 1)
queue_2.set_property("leaky", 2)

overlay = Gst.ElementFactory.make("textoverlay")

overlay.set_property("font-desc", "Sans, 32")

labels = load_labels(args["label_file"])

appsink.connect("new-sample", new_sample_callback,
                (interpreter, overlay, labels))

scale = Gst.ElementFactory.make("videoscale")

caps_str = "video/x-raw, width=(int){0}, height=(int){1}, format=(string){2}".format(
    width, height, "RGB")

appsink_caps = Gst.Caps.from_string(caps_str)
appsink.set_property("caps", appsink_caps)

conv_caps_str = "video/x-raw,width={}, height={}".format(
    video_width, video_height)

conv_caps = Gst.Caps.from_string(conv_caps_str)
conv_filter = Gst.ElementFactory.make("capsfilter")
conv_filter.set_property("caps", conv_caps)

if not src or not camera_filter or not nv_vid_conv or not h_264_enc \
        or not rtp_264_pay or not tee or not queue_1 or not queue_2 or not udp_sink \
        or not overlay or not videoconvert_1 or not scale or not appsink \
        or not conv_filter:
    print("Not all elements could be created.")
    exit(-1)


#Add all elements to the current pipeline

pipeline = Gst.Pipeline()

pipeline.add(src)
pipeline.add(camera_filter)
pipeline.add(nv_vid_conv)
pipeline.add(h_264_enc)
pipeline.add(rtp_264_pay)
pipeline.add(tee)
pipeline.add(queue_1)
pipeline.add(queue_2)
pipeline.add(udp_sink)
pipeline.add(overlay)
pipeline.add(videoconvert_1)
pipeline.add(scale)
pipeline.add(appsink)
pipeline.add(conv_filter)

#Link all GST elements together

if not Gst.Element.link(src, camera_filter):
    print("1 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(camera_filter, nv_vid_conv):
    print("2 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(nv_vid_conv, conv_filter):
    print("3 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(conv_filter, tee):
    print("4 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(tee, queue_2):
    print("5 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(queue_2, overlay):
    print("6 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(overlay, h_264_enc):
    print("7 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(h_264_enc, rtp_264_pay):
    print("8 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(rtp_264_pay, udp_sink):
    print("9 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(tee, queue_1):
    print("10 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(queue_1, scale):
    print("11 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(scale, videoconvert_1):
    print("12 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(videoconvert_1, appsink):
    print("13 Elements could not be linked.")
    exit(-1)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        pass

except KeyboardInterrupt:

    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)
