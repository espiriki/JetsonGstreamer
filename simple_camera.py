from gi.repository import Gst, GObject
import argparse
import sys
import gi
import traceback
gi.require_version('Gst', '1.0')

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


# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)

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
    "video/x-raw(memory:NVMM),width=1280, height=720, framerate=120/1, format=NV12")
camera_filter = Gst.ElementFactory.make("capsfilter", "filter")
camera_filter.set_property("caps", camera_caps)

nv_vid_conv = Gst.ElementFactory.make("nvvidconv")
nv_vid_conv.set_property("flip-method", flip_values[args["flip"]])

h_264_enc = Gst.ElementFactory.make("omxh264enc")

rtp_264_pay = Gst.ElementFactory.make("rtph264pay")
rtp_264_pay.set_property("config-interval", 1)
rtp_264_pay.set_property("pt", 96)

udp_sink = Gst.ElementFactory.make("udpsink")
udp_sink.set_property("host", args["ip_addr"])
udp_sink.set_property("port", 5000)

pipeline.add(src)
pipeline.add(camera_filter)
pipeline.add(nv_vid_conv)
pipeline.add(h_264_enc)
pipeline.add(rtp_264_pay)
pipeline.add(udp_sink)

src.link(camera_filter)
camera_filter.link(nv_vid_conv)
nv_vid_conv.link(h_264_enc)
h_264_enc.link(rtp_264_pay)
rtp_264_pay.link(udp_sink)

# https://lazka.github.io/pgi-docs/Gst-1.0/classes/Bus.html
bus = pipeline.get_bus()

# allow bus to emit messages to main thread
bus.add_signal_watch()

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Init GObject loop to handle Gstreamer Bus Events
loop = GObject.MainLoop()

# Add handler to specific signal
# https://lazka.github.io/pgi-docs/GObject-2.0/classes/Object.html#GObject.Object.connect
bus.connect("message", on_message, loop)

try:
    loop.run()
except Exception:
    pipeline.set_state(Gst.State.NULL)
    loop.quit()

# Stop Pipeline
pipeline.set_state(Gst.State.NULL)
