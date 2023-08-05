import sys
import argparse

from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput, Log

parser = argparse.ArgumentParser(description="Input image of your injury",formatter_class=argparse.RawTextHelpFormatter, epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
try:
    args = parser.parse_known_args()[0]
except:
    parser.print_help()
    sys.exit(0)
#print("Where is your injury located?")
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
net = detectNet(model="ssd-mobilenet.onnx", labels="labels.txt", 
                input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                threshold=args.threshold)
flag = True
while flag:
    img = input.Capture()
    if img is None:
        continue  
    detections = net.Detect(img, overlay=args.overlay)
    #print("detected {:d} objects in image".format(len(detections)))
    for detection in detections:
        classLabel = net.GetClassDesc(detection.ClassID)
        #print(detection)
        if classLabel == "Human head":
            print("You have injured your head! Your head/brain area is very delicate, so it is important to see a specialist right away to check for a concussion or an underlying iusue")
            flag = False
        elif classLabel == "Human leg":
            print("You have hurt your leg! Your legs are extremely vital to your body, so the best thing is to go see a doctor right away")
            flag = False
        elif classLabel == "Human hand":
            print("You have hurt your hand! The most common causes of hand or arm injury are blunt trauma, broken bones, or high-pressure impact. You can ice it, elevate, and most importantly, rest!")
            flag = False
    output.Render(img)
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
    net.PrintProfilerTimes()
    if not input.IsStreaming() or not output.IsStreaming():
        break