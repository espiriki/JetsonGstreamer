Gstreamer TFLite Demo

Runs a gstreamer pipeline that streams the video over UDP and set the label
of the image using TFLite inference engine


To run:

    python3 gstreamer_label_jetson.py --ip 127.0.0.1 --m mobilenet_v1_1.0_224.tflite --l mobilenet_v1_1.0_224/labels.txt --flip 180 --port 5000

To watch the stream:

    gst-launch-1.0.exe -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! queue leaky=1 ! decodebin ! videoconvert ! autovideosink sync=false