# opencv_yolov5
using opencv run yolov5 .onnx format file

yolov5=6.1 opencv=4.5.5

python export.py --weights yolov5s.pt --simplify --include onnx
