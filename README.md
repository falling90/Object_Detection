# Object Detection Algorithms

>**발전 과정**

    * R-CNN → SPPnet → Fast R-CNN → Faster R-CNN → YOLO(v1 to v5)   
   

## R-CNN
>**R-CNN Algorithms**

    - 
    
	
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/R-CNN/1.PNG" width="800px" height="500px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/R-CNN/2.PNG" width="800px" height="500px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/R-CNN/3.PNG" width="800px" height="500px"></img><br/>  


>**R-CNN 단점**

    - Object Detection 속도 자체가 느림.
    - Selective Search 를 통한 검출된 Region Proposals 마다 CNN을 적용하기때문에 시간 多 소요.
    - 합성곱 신경망(CNN) 입력을 위해 고정된 크기로 변환(Warp/Crop) 하는 과정에서 Image 정보 손실 발생
    - 학습이 여러 단계로 이루어져 긴 학습시간과 대용량 저장공간 필요함.


## SPPnet
>**SPPnet Algorithms**

    - SPP 활용을 통해 R-CNN의 느린 속도 개선(학습 : 3배, 실적용 : 10~100배)
    - R-CNN과 같은 구조로 여러 학습 단계가 적용되어야 하고 대용량 저장 공간 필요함.

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/SPPnet/1.PNG" width="800px" height="500px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/SPPnet/2.PNG" width="800px" height="500px"></img><br/>  

>**R-CNN vs SPPnet**

    - SPP(Spatial Pyramid Pooling)을 통해 합성곱 신경망(CNN) 계산을 한번만 한다.

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/SPPnet/3.PNG" width="800px" height="500px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/SPPnet/4.PNG" width="800px" height="500px"></img><br/>  


### Main Results
![plot](https://github.com/falling90/Object_Detection/blob/main/Result/Result1.png?raw=true)
![plot](https://github.com/falling90/Object_Detection/blob/main/Result/Result2.png?raw=true)
![plot](https://github.com/falling90/Object_Detection/blob/main/Result/Result3.png?raw=true)

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
