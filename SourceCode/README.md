# Object Detection API : ssd_mobilenet_v1_coco

SSD : Single Shot MultiBox Detector

MobileNet : Object Detector released in 2017 as an efficient CNN architecture designed for mobile and embedded vision application

### Introduction

**SSD_mobilenet_v1_coco** was described in [Here](https://docs.openvinotoolkit.org/latest/omz_models_model_ssd_mobilenet_v1_coco.html).

This code has been tested on Windows 10 64-bit, and on Colaboratory.

It initially described in [Here](https://github.com/tensorflow/models/tree/991f75e200721267302291862cd9bf936ca06f90/research/object_detection).

Model is available at [Here](https://github.com/tensorflow/models).

TestData is available at [Here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz).

It doesn't included any training data (Trained Model included in TestData)

### Main Results
![plot](https://github.com/falling90/Object_Detection/blob/main/SourceCode/Result/Result1.png?raw=true)
![plot](https://github.com/falling90/Object_Detection/blob/main/SourceCode/Result/Result2.png?raw=true)
![plot](https://github.com/falling90/Object_Detection/blob/main/SourceCode/Result/Result3.png?raw=true)

### Contents

1. [Requirements: software](#requirements-software)
2. [Testing Demo](#testing-demo)

### Requirements: software

1. `Tensorflow` build for Object Detection
    - Install tensorflow=="2.*"
    - Install tf_slim
    - Install pycocotools

### Testing Demo:
1.	Detection_model
    - Inputs
	```Shell
	[<tf.Tensor 'image_tensor:0' shape=(None, None, None, 3) dtype=uint8>]
	```

    - Output Dtypes
	```Shell
	detection_boxes': tf.float32,
	detection_classes': tf.float32,
	detection_scores': tf.float32,
	num_detections': tf.float32
	```

    - Output Shapes
	```Shell
	detection_boxes': TensorShape([None, 100, 4],
	detection_classes': TensorShape([None, 100]
	detection_scores': TensorShape([None, 100],
	num_detections': TensorShape([None]
	```

    - Label Category

	|                |                |                |                |                |
	| :-------------:| :-------------:| :-------------:| :-------------:| :-------------:|
	| Airplane       | Apple          | Backpack       | Banana         | Baseball bat   |
	| Baseball glove | Bear           | Bed            | Bench          | Bicycle        |
	| Bird           | Boat           | Book           | Bottle         | Bowl           |
	| Broccoli       | Bus            | Cake           | Car            | Carrot         |
	| Cat            | Cell phone     | Chair          | Clock          | Couch          |
	| Cow            | Cup            | Dining table   | **Dog**            | Donut          |
	| Elephant       | Fire hydrant   | Fork           | Frisbee        | Giraffe        |
	| Hair drier     | Handbag        | Horse          | Hot dog        | Keyboard       |
	| **Kite**           | Knife          | Laptop         | Microwave      | Motorcycle     |
	| Mouse          | Orange         | Oven           | Parking meter  | **Person**         |
	| Pizza          | Potted plant   | Refrigerator   | Remote         | Sandwich       |
	| Scissors       | Sheep          | Sink           | Skateboard     | Skis           |
	| Snowboard      | Spoon          | Sports ball    | Stop sign      | Suitcase       |
	| Surfboard      | Teddy bear     | Tennis racket  | Tie            | Toaster        |
	| Toilet         | Toothbrush     | Traffic light  | Train          | Truck          |
	| TV             | Umbrella       | Vase           | Wine glass     | Zebra          |
