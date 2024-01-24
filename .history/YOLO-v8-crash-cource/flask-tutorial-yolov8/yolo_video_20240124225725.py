{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n",
    "import cv2\n",
    "import math\n",
    "\n",
    "def video_detection(path_x):\n",
    "    video_capture = path_x\n",
    "    #Create a Webcam Object\n",
    "    cap=cv2.VideoCapture(video_capture)\n",
    "    frame_width=int(cap.get(3))\n",
    "    frame_height=int(cap.get(4))\n",
    "    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))\n",
    "\n",
    "    model=YOLO(\"../yolo-weights/yolov8n.pt\")\n",
    "    classNames = [\"person\", \"bicycle\", \"car\", \"motorbike\", \"aeroplane\", \"bus\", \"train\", \"truck\", \"boat\",\n",
    "                  \"traffic light\", \"fire hydrant\", \"stop sign\", \"parking meter\", \"bench\", \"bird\", \"cat\",\n",
    "                  \"dog\", \"horse\", \"sheep\", \"cow\", \"elephant\", \"bear\", \"zebra\", \"giraffe\", \"backpack\", \"umbrella\",\n",
    "                  \"handbag\", \"tie\", \"suitcase\", \"frisbee\", \"skis\", \"snowboard\", \"sports ball\", \"kite\", \"baseball bat\",\n",
    "                  \"baseball glove\", \"skateboard\", \"surfboard\", \"tennis racket\", \"bottle\", \"wine glass\", \"cup\",\n",
    "                  \"fork\", \"knife\", \"spoon\", \"bowl\", \"banana\", \"apple\", \"sandwich\", \"orange\", \"broccoli\",\n",
    "                  \"carrot\", \"hot dog\", \"pizza\", \"donut\", \"cake\", \"chair\", \"sofa\", \"pottedplant\", \"bed\",\n",
    "                  \"diningtable\", \"toilet\", \"tvmonitor\", \"laptop\", \"mouse\", \"remote\", \"keyboard\", \"cell phone\",\n",
    "                  \"microwave\", \"oven\", \"toaster\", \"sink\", \"refrigerator\", \"book\", \"clock\", \"vase\", \"scissors\",\n",
    "                  \"teddy bear\", \"hair drier\", \"toothbrush\"\n",
    "                  ]\n",
    "    while True:\n",
    "        success, img = cap.read()\n",
    "        results=model(img,stream=True)\n",
    "        for r in results:\n",
    "            boxes=r.boxes\n",
    "            for box in boxes:\n",
    "                x1,y1,x2,y2=box.xyxy[0]\n",
    "                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)\n",
    "                print(x1,y1,x2,y2)\n",
    "                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)\n",
    "                conf=math.ceil((box.conf[0]*100))/100\n",
    "                cls=int(box.cls[0])\n",
    "                class_name=classNames[cls]\n",
    "                label=f'{class_name}{conf}'\n",
    "                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]\n",
    "                print(t_size)\n",
    "                c2 = x1 + t_size[0], y1 - t_size[1] - 3\n",
    "                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled\n",
    "                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)\n",
    "\n",
    "        yield img\n",
    "        #out.write(img)\n",
    "        #cv2.imshow(\"image\", img)\n",
    "        #if cv2.waitKey(1) & 0xFF==ord('1'):\n",
    "            #break\n",
    "    #out.release()\n",
    "cv2.destroyAllWindows()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mlproj",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
