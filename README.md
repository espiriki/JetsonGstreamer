Gstreamer TFLite Demo

Runs a gstreamer pipeline that streams the video over UDP and set the label
of the image using TFLite inference engine

The --flip argument is used to flip the video (steps of 90 degrees)

To run (change 127.0.0.1 to the IP of your PC):

    python3 gstreamer_label_jetson.py --ip 127.0.0.1 --m final_it_EfficientNetB0_52_96_percent_v2_best_model.h5.tflite --flip 180 --port 5000

To watch the stream:

    gst-launch-1.0.exe -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! queue leaky=1 ! decodebin ! videoconvert ! autovideosink sync=false