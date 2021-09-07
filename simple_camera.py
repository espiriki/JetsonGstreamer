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

flip_values = {
    "0": 0,
    "90": 1,
    "180": 2,
    "270": 3
}


def on_message(bus: Gst.Bus, message: Gst.Message, loop: GObject.MainLoop):
    mtype = message.type
    """
        Gstreamer Message Types and how to parse
        https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
    """
    if mtype == Gst.MessageType.EOS:
        print("End of stream")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(err, debug)
        loop.quit()

    elif mtype == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(err, debug)

    return True


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def setup_model(args, interpreter):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args["image"]).resize((width, height))

    print(img)

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:

        print("Float model!")

        input_data = (np.float32(input_data) -
                    args["input_mean"]) / args["input_std"]

    get_labels(interpreter, input_data, floating_model, height, width, False)

    return (width,height)


def get_labels(interpreter, input_data, floating_model, height, width, convert):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if convert:

        deserialized_bytes = np.frombuffer(input_data, dtype=np.uint8)
        
        deserialized_x = np.reshape(deserialized_bytes, newshape=(1, height, width, 3))

        deserialized_y = (np.float32(deserialized_x) - 127.5) / 127.5

        deserialized_z = deserialized_y.view('<f4')

        interpreter.set_tensor(input_details[0]['index'], deserialized_z)

    else:

        interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args["label_file"])

    stop_time = time.time()

    best_result = top_k[0]

    split_string =  labels[best_result].split(",", 1)[0]

    label =  split_string.split(":", 2)[1]

    score = float(results[best_result])
    
    return score, label

def new_sample_callback(appsink, interpreter):

    sample = appsink.emit("pull-sample")

    buffer = sample.get_buffer()

    caps_format = sample.get_caps().get_structure(0)

    frmt_str = caps_format.get_value('format')
    video_format = GstVideo.VideoFormat.from_string(frmt_str)

    w, h = caps_format.get_value('width'), caps_format.get_value('height')

    c = utils.get_num_channels(video_format)

    ret, map_info = buffer.map(Gst.MapFlags.READ)

    results = get_labels(interpreter[0], map_info.data, True, w, h, True)

    interpreter[1].set_property("text", "{} - {:.2f}%".format(results[1],results[0]*100))

    return Gst.FlowReturn.OK


Gst.init(None)

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
    '-i',
    '--image',
    required=True,
    help='image to be classified')

ap.add_argument(
    '--num_threads', default=1, type=int, help='number of threads')


ap.add_argument(
    '--input_mean',
    default=127.5, type=float,
    help='input_mean')

ap.add_argument(
    '--input_std',
    default=127.5, type=float,
    help='input standard deviation')

ap.add_argument(
    '-l',
    '--label_file',
    help='name of file containing labels',
    required=True)

args = vars(ap.parse_args())


interpreter = tflite.Interpreter(model_path=args["model_file"])

interpreter.allocate_tensors()

width, height = setup_model(args, interpreter)

# create pipeline object
pipeline = Gst.Pipeline()

# create Gst.Element by plugin name
src = Gst.ElementFactory.make("nvarguscamerasrc")
src.set_property("sensor_id", 0)

camera_caps = Gst.Caps.from_string(
    "video/x-raw(memory:NVMM),width=1280, height=720, framerate=59/1, format=NV12")
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

cairo = Gst.ElementFactory.make("cairooverlay")

overlay = Gst.ElementFactory.make("textoverlay")

overlay.set_property("font-desc", "Sans, 32")

appsink.connect("new-sample", new_sample_callback, (interpreter, overlay))

my_format = "RGB"

caps_str = "video/x-raw, width=(int){0}, height=(int){1}, format=(string){2}".format(width, height, my_format)

appsink_caps = Gst.Caps.from_string(caps_str)

appsink.set_property("caps", appsink_caps)

if not udp_sink or not rtp_264_pay or not videoconvert_1 or not videoconvert_2 \
        or not appsink or not h_264_enc or not nv_vid_conv or not camera_filter or not src \
        or not camera_caps or not tee or not queue_2 or not queue_1 or not cairo or not overlay:
    print("Not all elements could be created.")
    exit(-1)

pipeline.add(src)
pipeline.add(camera_filter)
pipeline.add(nv_vid_conv)
pipeline.add(h_264_enc)
pipeline.add(rtp_264_pay)
pipeline.add(appsink)
pipeline.add(tee)
pipeline.add(queue_1)
pipeline.add(queue_2)
pipeline.add(udp_sink)
pipeline.add(cairo)
pipeline.add(overlay)
pipeline.add(videoconvert_1)

if not Gst.Element.link(src, camera_filter):
    print("1 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(camera_filter, nv_vid_conv):
    print("2 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(nv_vid_conv, tee):
    print("3 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(tee, queue_1):
    print("4 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(queue_1, videoconvert_1):
    print("4444 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(videoconvert_1, appsink):
    print("1122 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(tee, queue_2):
    print("6 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(queue_2, overlay):
    print("7 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(overlay, h_264_enc):
    print("11 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(h_264_enc, rtp_264_pay):
    print("8 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(rtp_264_pay, udp_sink):
    print("9 Elements could not be linked.")
    exit(-1)


# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

try:

    while True:
        pass    

except KeyboardInterrupt:
    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)


