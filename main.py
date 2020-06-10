"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-tpt", "--total_people_threshold", type=float, default=10,
                        help="Threshold for number of people counted")
    
    return parser


def draw_boxes(frame, result,prob_threshold,initial_w,initial_h):
    '''
    Draw bounding box for object when it exceeds it's probability threshold
    :param frame: frame from camera/video
    :param result: list contains the data comming from inference
    :return: person count and frame
    '''
    count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_width)
            ymin = int(obj[4] * initial_height)
            xmax = int(obj[5] * initial_width)
            ymax = int(obj[6] * initial_height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            count += 1
            
    return frame, count


def connect_mqtt():
    ### Connecting to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Inititializing some variables
    counter = 0
    present = 0
    last_count = 0
    prev_duration = 0
    total_count = 0
    duration = 0
    current_request_id = 0
    duration_report = None
        
    # Flag for input image
    single_image_mode = False
    
    # Initialise the class
    infer_network = Network()
    infer_network.load_model(args.model,args.device,1,1,current_request_id,args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "input file doesn't exist"
    
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
    
    if not cap.isOpened():
        log.error("Error! Unable to open video source")
        
    global prob_threshold, total_people_threshold, initial_width, initial_height
    prob_threshold = args.prob_threshold
    total_people_threshold = args.total_people_threshold
    initial_width = cap.get(3)
    initial_height = cap.get(4)

    ### TODO: Loop until stream is over ###  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        key_pressed = cv2.waitKey(60)
        
        w, h = net_input_shape[3], net_input_shape[2]
        # Change data layout from HWC to CHW
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((1,*image.shape))
        
        # Start asynchronous inference for specified request.
        inf_start = time.time()
        infer_network.exec_net(current_request_id, image)
        
        duration_report = None
        # Wait for the result
        if infer_network.wait(current_request_id) == 0:
            det_time = time.time() - inf_start
            
            # Results of the output layer of the network
            result = infer_network.get_output(current_request_id) 
            frame_with_box, current_count = draw_boxes(frame, result, prob_threshold, initial_width, initial_height)
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame_with_box, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    
            if current_count != counter:
                last_count = counter
                counter = current_count
                if duration >= 3:
                    prev_duration = duration
                    duration = 0
                else:
                    duration += prev_duration
                    prev_duration = 0
            else:
                duration += 1
                if duration >= 3:
                    present = counter
                    if duration == 3 and counter > last_count:
                        total_count += counter - last_count
                    elif duration == 3 and counter < last_count:
                        duration_report = int((prev_duration / 10.0) * 1000)            
                        
            client.publish('person',
                           payload=json.dumps({
                               'count': present, 'total': total_count}),
                           qos=0, retain=False)
            
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
                
            if total_count > total_people_threshold:
                client.publish('peopleThreshold',
                               payload=json.dumps({'peopleThreshold': total_count}),
                               qos=0, retain=False)
                

            if key_pressed == 27:
                break

        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame_with_box)  
        sys.stdout.flush()
    
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame_with_box)
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client)
    
    
if __name__ == '__main__':
    main()