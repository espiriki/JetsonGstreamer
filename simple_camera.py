from gi.repository import Gst, GObject
import argparse
import sys
import gi
import traceback
gi.require_version('Gst', '1.0')


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


def add_IP_to_pipeline(default_pipeline, ip_addr):

    return default_pipeline.replace("127.0.0.1", ip_addr)


# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)

DEFAULT_PIPELINE = "nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM),width=1280, height=720, framerate=120/1, format=NV12" +\
    "! omxh264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5000"

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

ap.add_argument("-ip", "--ip_addr", required=True,
                default="127.0.0.1", help="IP to stream to")

args = vars(ap.parse_args())

command = args["pipeline"]

command = add_IP_to_pipeline(command, args["ip_addr"])

print(command)

# Gst.Pipeline https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html
# https://lazka.github.io/pgi-docs/Gst-1.0/functions.html#Gst.parse_launch
pipeline = Gst.parse_launch(command)

# sys.exit(1)

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
