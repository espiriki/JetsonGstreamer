from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GObject
import argparse
import sys
import traceback

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

def new_sample_callback(appsink, udata):
    sample = appsink.emit("pull-sample")
    print("new_buffer")
    print(sample)
    return Gst.FlowReturn.OK


Gst.init(None)

ap = argparse.ArgumentParser()

ap.add_argument("-ip", "--ip_addr", required=True,
                default="127.0.0.1", help="IP to stream to")

ap.add_argument("-flip", "--flip", required=False,
                default="0", help="video flip in degrees")

args = vars(ap.parse_args())

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

appsink = Gst.ElementFactory.make("appsink","sink")
appsink.set_property("emit-signals", True)
appsink.connect("new-sample", new_sample_callback, appsink)

videoconvert_1 = Gst.ElementFactory.make("nvvidconv")
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

if not udp_sink or not rtp_264_pay or not videoconvert_1 or not videoconvert_2 \
 or not appsink or not h_264_enc or not nv_vid_conv or not camera_filter or not src \
 or not camera_caps or not tee or not queue_2 or not queue_1 or not cairo:
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

if not Gst.Element.link(queue_1, appsink):
    print("5 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(tee, queue_2):
    print("6 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(queue_2, h_264_enc):
    print("7 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(h_264_enc, rtp_264_pay):
    print("8 Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(rtp_264_pay, udp_sink):
    print("9 Elements could not be linked.")
    exit(-1)        

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

while True:
    pass

# Stop Pipeline
pipeline.set_state(Gst.State.NULL)
