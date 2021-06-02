# Object Detection Algorithms

>**발전 과정**

    -. R-CNN → SPPnet → Fast R-CNN → Faster R-CNN → YOLO(v1 to v5) → PP-YOLO
   

## R-CNN
>**R-CNN Algorithms**

    1. Segmentation을 통해 초기 Region 생성(유사한 색상으로 매칭)
    2. Selective Search 을 통해 약 2,000 여개의 Region Proposal 진행
    3. 각 Region 別 Crop & Warp 진행(※ CNN 입력을 위해서 고정된 크기로 변환 필요)
    4. 각각 CNN 적용을 통해 Feature Map 생성
    5. 추출된 Feature Map 은 Classifiers(SVM)을 통해 분류한다.
    
    * 학습
        - Input으로 Image와 
        - 세 단계로 진행 (Conv Fine Tune → SVM Classification → BB Regression)

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/1.R-CNN/1.PNG" width="800px" height="300px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/1.R-CNN/2.PNG" width="800px" height="400px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/1.R-CNN/3.PNG" width="800px" height="300px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/1.R-CNN/4.PNG" width="800px" height="500px"></img><br/>  

>**R-CNN 단점**

    -. Object Detection 속도 자체가 느림.
    -. Selective Search 를 통한 검출된 Region Proposals 마다 CNN을 적용하기때문에 시간 多 소요.
    -. 합성곱 신경망(CNN) 입력을 위해 고정된 크기로 변환(Warp/Crop) 하는 과정에서 Image 정보 손실 발생
    -. 학습이 여러 단계로 이루어져 긴 학습시간과 대용량 저장공간 필요함.


## SPPnet
>**SPPnet Algorithms**

    -. 이미지를 통해 Conv Feature Map 생성
    -. Spatial Pyramid Pooling(SPP) 활용을 통해 R-CNN의 느린 속도 개선(학습 : 3배, 실적용 : 10~100배)
    -. R-CNN과 같은 구조로 여러 학습 단계가 적용되어야 하고 대용량 저장 공간 필요함.

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/2.SPPnet/1.PNG" width="800px" height="500px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/2.SPPnet/2.PNG" width="800px" height="300px"></img><br/>  

>**R-CNN vs SPPnet**

    -. Spatial Pyramid Pooling(SPP)을 통해 합성곱 신경망(CNN) 계산을 한번만 한다.

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/2.SPPnet/3.PNG" width="800px" height="100px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/2.SPPnet/4.PNG" width="1000px" height="500px"></img><br/>  

## Fast R-CNN
>**Fast R-CNN Algorithms**

    -. R-CNN & SPPnet 대비 성능 개선 (학습시간 : 9배 / 3배,  실적용 : 213배 / 10배)
    -. ROI Pooling Layer를 통해 고정된 Feature Vector 생성
    -. Single-Stage로 학습 진행
    -. 전체 네트워크 업데이트 가능
    -. 저장 공간 불필요

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/1.PNG" width="800px" height="300px"></img><br/>  

    * ROI Pooling Layer
      > ROI 영역에 해당하는 부분만 Max Pooling을 통해 Feature Map으로부터 고정된 길이의 저차원 벡터로 축소
        - 각각의 ROI 는 (r, c, h, w)의 튜플 형태 [(r, c) : Top-Left Corner의 좌표]
        - h*w ROI 사이즈를 H*W의 작은 윈도우 사이즈로 나눔 (h/H * w/W)
        - SPPnet의 SPP Layer의 한 Pyramid Level 만 사용하는 형태는 동일
        
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/2.png" width="800px" height="150px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/3.png" width="800px" height="300px"></img><br/>  

    * 학습 : 기본적으로 ImageNet을 사용한 Pre-Trained Network Base로 진행
    * 변경점
        - 마지막 Max Pooling Layer → ROI Pooling Layer로 대체(VGG16에서는 H=W=7)
        - 신경망의 마지막 FC Layer와 Softmax단이 두 개의 Output Layer로 대체(ImageNet : 1000개 분류)
        - 신경망의 입력이 Image와 ROI를 반영할 수 있도록 변경

>**SPPnet vs Fast R-CNN**
    
    -. Region-wise Sampling → Hierarchical Sampling
        - SGD 미니배치가 Image에 종속됨(N개의 이미지를 각 R/N ROI개로)
        - N=2, R=128로도 좋은 학습 결과 확인
        - 약 64배 빠른 학습 가능(N 이 작을수록 계산복잡도 낮아짐)
    -. SPP Layer → ROI Pooling Layer
    -. Single-Stage(Multi-Task)
        - 최종 Classifier와 Regression까지 단방향 단계 [→ 효율적인 학습 (Softmax Classifier + Bounding Box Regressor)]

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/4.png" width="1000px" height="500px"></img><br/>  

>**Fast R-CNN 한계**
    
    -. Region Proposal 단계가 대부분의 시간을 차지하기 때문에 Testing 단계에서 여전히 느림.

## Faster R-CNN
>**Faster R-CNN Algorithms**

    -. Region Proposal에 신경망을 적용[RPN(Region Proposal Network)]하여 시간 단축 (CPU → GPU)
    -. Region-Based 감지기의 Feature Map을 Region-Proposal Generating 에도 사용
      → 전체 Image를 Convolutional Network 하여 만들어낸 Feature Map을 Region-Proposal Generating 에서 활용
    -. RPN은 End to End로 학습 가능
    -. Object여부와 Bounding Box를 Regress하는 하나의 FCN
    
    ※ 어텐션처럼 RPN은 Fast R-CNN이 어디에 주목해야하는지 알려준다.

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/1.PNG" width="800px" height="300px"></img><br/>  

    * RPN (Region Proposal Networks)
        - 다양한 사이즈의 Image를 입력값으로 Object Score 와 Object Proposal을 출력
          → 만들어낸 Proposal이 Object인지? 아닌지?
          → 실제 Ground Truth 에 맞춰서 좌표값을 Regression 할수있는지 학습하는 Fully Connected Network
        - Fast R-CNN과 합성곱 신경망을 공유
        - Feature Map의 마지막 Conv 층을 작은 네트워크가 Sliding하여 저차원으로 Mapping (Anchor Box 사용)
        
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/2.PNG" width="800px" height="300px"></img><br/>  

    * Anchor Box
        - 각 Sliding Window 에서 Bounding Box의 후보로 사용되는 상자
        - Regressoin층 = Anchor Box * 4,  Classification층 = 2k (→ k = Object인지 아닌지를 의미)
        - Anchor는 Sliding Window 중심에 있고, 미리 정해진 Scale과 비율에 따라 달라진다.
        - 실험에서는 3개의 Scale, 3개의 비율을 사용하여 k=9개의 앵커 사용.
        - W*H크기만큼의 Conv Feature Map에서 WHk만큼의 앵커가 있는 것이다.

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/3.PNG" width="800px" height="300px"></img><br/>  

    * NMS(Non Maximum Suppression) : 가장 확실한 박스 선정
        - 생성된 Anchor Box 에서 NMS을 사용하여 약 2000개의 ROI 생성
        - 2000개의 후보 영역 중 Sampling을 통해 Proposal을 뽑아 Fast R-CNN Detector로 보내 학습 진행

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/4.PNG" width="300px" height="250px"></img><br/>  

    * Translation-Invariant(이동불변성) Anchors
        - RPN에서 Window Size를 Sliding 하는 방식은 이동 불변성 보장
        - CNN에서 Window Sliding을 사용하여 Convolution 하였을때 얻는 효과와 동일
        - Model Size를 줄여준다.((4+1)*800 vs (4+2)*9) → 연산량 절감

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/5.PNG" width="800px" height="200px"></img><br/>  

    * Multi-Scale Anchors as Regression References
        - Anchors Box의 Hyper Parameter를 다양하게 둠으로써, 다양한 이미지 or 필터를 사용한 것과 동일한 효과
        - Pyramid of Anchors
        - 다양한 Scale과 Ratio를 활용한 Anchor를 통해 효율적으로 계산 가능

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/6.PNG" width="800px" height="200px"></img><br/>  

    * 학습
        1) RPN은 Imagenet을 사용하여 학습된 모델로부터 초기화되고, Region Proposal Task를 위해 End to End 학습
        2) 위 단계에서 학습된 RPN을 사용하여 Fast R-CNN 모델의 학습을 진행한다.(초기화는 ImageNet 학습 모델로)
        3) RPN을 다시 한번 학습하는데 공통된 Conv Layer(Fully Convolutional Features)는 고정하고 RPN에만 연결된 층만 학습
        4) 공유된 Conv Layer를 고정시키고, Fast R-CNN을 다시 학습한다.
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/7.png" width="800px" height="300px"></img><br/>  


>**Fast R-CNN vs Faster R-CNN**
    
    -. RPN(Region Proposal Network)를 통해 Cost-Free한 Region Proposal 방법론 제안
    -. 빠른 속도와 높은 정확도의 Object Detection을 이루어냄.

## R-CNN OverView
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/9.png" width="1000px" height="250px"></img><br/>  


## YOLO v1
    -. YOLO : You Only Look Once

>**YOLO v1 특징**

    -. Image로부터 한 번에 Class와 Bounding Box를 예측(Fast R-CNN)
    -. Inference가 굉장히 빠름
    -. 전체 이미지를 보고 Object Detection을 수행하기 때문에 배경 오류가 적고 일반화 성능이 좋음
    -. 성능이 낮음(특히 Small Object를 잘 탐지하지 못함

    -. 각 GridCell 마다 정답 Vector를 생성하여야 한다.
    -. 정답 Vector는 2개의 Bounding Box 를 사용.
    -. 각 Bounding Box는 5개의 정보 [Confidence Score(1개) + Bounding Box 정보(4개)]로 이루어져있음.
      → Confidence Score : Bounding Box 內 Object가 포함되어있을 확률
      → Bounding Box 정보 : B/Box 가운데 좌표 2개(x, y) + 크기 정보 2개(width, height)

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/5.YOLO_v1/1.PNG" width="800px" height="300px"></img><br/>  

>**YOLO v1 Training**

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/5.YOLO_v1/2.png" width="800px" height="500px"></img><br/>  
    
>**YOLO v1 Inference**

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/5.YOLO_v1/3.png" width="800px" height="500px"></img><br/>  

>**YOLO v1 단점**

    -. 각 Grid Cell마다 2개의 경계 상자를 예측하고 1개의 Class 밖에 가질 수 없기 때문에 가깝게 붙어 있는 물체를 잘 탐지하지 못함.
    -. 작은 물체를 잘 탐지하지 못하고 Recall 이 낮음

## YOLO v2
    -. Recall을 올리고 Localization 개선
    -. Better and Faster!

>**YOLO v2 특징**

  * 성능 향상(Better)

        1) Batch Normalization
          → 모든 Conv Layer에 Batch Norm 적용
          → Batch Norm을 통해 Regularization 효과를 얻어 Drop-out 제거
          → mAP 기준 약 2%의 성능 향상을 얻음
    
        2) High Resolution Classifier
          → Yolo v1 = Detection Task의 절반의 해상도로(224 by 224) Classifier 학습
          → Yolo v2 = Detection Task와 같은 해상도로 Classifier 학습
          → 약 4% 의 성능 향상을 얻음
      
        3) Convolutional Anchor Boxes
          → FC Layer를 삭제하고 Fully Convolution 구조 사용
          → Anchor Box를 활용해 경계 상자를 예측
          → 1개의 Center를 갖는 13 by 13 output feature map을 만들기 위해 input image 크기를 416 by 416 으로 조정함
          → Yolo v1은 7*7*2 총 98개의 경계 상자를 예측한 반면 Anchor box를 사용하는 v2는 13*13*5개의 경계 상자를 도출.
          → mAP는 소폭 감소(69.5 → 69.2)하였지만, Recall이 크게 향상(81% → 88%)됨.
      
        4) Dimention Clusters
          → 미리 정의한 Anchor box를 사용하지 않고 데이터에 맞는 Anchor box를 사용
          → 데이터에 존재하는 Ground truth box를 이용하여 Clustering 실시
          → 최대한 Ground truth box와 유사한(IOU가 높은) Anchor box를 찾는 것을 목적으로 함
          → 총 5개의 Anchor box를 사용
            
        5) Direct Location Prediction
          → (x, y)를 특정 Grid Cell 안으로 한정하여 학습 초반 Random Initialization으로 인한 학습의 불안정성을 예방
          → Dimension Cluster와 Direct location prediction을 사용한 결과 약 5%의 성능 향상을 얻음
      
        6) Fine-Grained Features
          → 작은 물체를 잘 탐지하기 위해 더 높은 해상도를 가진 이전 단계의 Layer를 가져와
            Detection을 위한 Output feature map에 Concatenate하여 사용
          → Feature Map 변환 후 사용 ([26 by 26 by 512] → [13 by 13 by 2048]
          → 약 1% 성능 향상을 얻음

        7) Multi-Scale Training
          → 매 10 batch 마다 Input image의 크기를 바꿔가며 모델 학습
          → 32씩 크기를 증가시킨 {320, 352, ..., 608}에 대해 학습을 진행함
          → 이를 통해 모델이 Input size에 대해 강건하게 함.
      
  * 속도 성능 (Faster)   

        -. DarkNet
          → VGG-16 기반의 Detection Frameworks는 충분히 좋은 성능을 보이지만 과도하게 복잡하여 DarkNet이라는 새로운 네트워크 구조를 제안
          → 최종 모델로 19개의 Convolutional Layer와 5개의 Maxpooling Layer를 가진 DarkNet-19를 사용함
            [(VGG-16 → DarkNet-19) ==> (30.69 → 5.58 (Billion Operation))]
          → ImageNet 데이터에 대해 72.9%의 Top-1 Accuracy와 91.2% Top-5 Accuracy를 보임
      

  * Yolo 9000-Stronger 구조

        -. Classification Task 데이터 셋을 활용하여 좀 더 다양한 Class를 예측할 수 있는 Detection Model을 학습할 수 있는 방법

        -. Classification Task 데이터 셋의 Class와 Detection Task 데이터 셋의 Class는 Mutually Exclusive 하지 않음
        -. 따라서 WordNet의 구조를 트리 형태로 변형하여 사용
        -. Object Detection에서 모든 개념을 포괄하는 Physical Object를 Root Node로 설정
        -. 자손 Node로부터 조상 Node까지의 경로가 2개 이상인 경우 가장 짧은 경로를 채택
        
        
  * Yolo 9000-Stronger 실험(Classification)

        -. 1000개의 Class를 갖는 이미지넷 데이터셋을 통해 WordTree를 구축하고 DarkNet을 활용하여 실험 진행
        -. Tree 구축 후 만약 자손 Node의 Label 값이 1이면 모든 조상 Node의 Label 값도 1로 바꾸어 실험 진행
        -. Classification은 아래 그림과 같이 Multiple Softmax 방식 활용

        -. 결과 : 369개의 새로운 Class가 추가되었음에도, 1% 미만의 성능 차이를 보임


  * Yolo 9000-Stronger 실험(Object Detection)

        -. COCO Detection 데이터 셋과 이미지넷의 상위 9000개 Class에 대한 데이터를 통해 WordTree를 구축한 후 Yolo v2 학습
        -. 총 9418개의 Class를 갖는 WordTree를 구축하였고, COCO 데이터를 Oversampling하여 이미지넷 데이터 수와의 비율을 4:1로 맞춤
        -. 실험에는 Output 크기를 줄이기 위해 3개의 Anchor Box를 사용
        -. Input이 Classification 데이터인 경우 단순히 그 Class에 대한 예측 값이 가장 높은 경계 상자에 대해 Classification Loss를 계산하여 역전파 실시
        -. 모델의 평가를 위해 이미지넷의 Detection Task 데이터셋을 활용
        -. 이 데이터는 COCO 데이터셋과 오직 44개의 Class만을 공유하며 나머지 156개의 Class는 COCO 데이터 셋에 존재하지 않음

        -. 결과 : 전체 Class에 대해 19.7 mAP의 성능을 보였고, COCO 데이터셋에 존재하지 않는 156개의 Class에 대해서는 16.0 mAP의 성능을 보임
                  학습데이터에 포함되지 않은 데이터로 성능을 평가했음에도 DPM 보다 성능 우수
        -. COCO Detection 데이터셋은 많은 동물 데이터를 포함하고 있어,
           동물과 관련된 새로운 Class에 대해서는 성능이 좋은 반면, 다른 Class에 대해서는 매우 안좋은 성능을 보임

## YOLO v3
    -. 조금 느리지만 성능이 더 높은 모델을 만들어보자!
    
>**YOLO v3 특징**

    1) Bounding Box & Class Prediction
      → 역전파 계산식 L2 Loss → L1 Loss 로 변경
      → 각 Class에 대해 독립적으로 Logistic Regression을 적용하고,
        Binary Cross Entropy로 Loss를 계산함
      → 이로 인해 상호 연관된 Class에 대해서 더 잘 예측하고, 하나의 Object에 대해 여러 Clss를 예측함
      
    2) Predictions Across Scales
      → 3개의 다른 크기의 Feature마다 각각 Prediction을 진행함.
      → 첫 Prediction 이후 더 큰 Feature Map에 대한 Prediction을 위해
        Up-Sampling과 이전단계의 Layer를 가져와 Concatenation 하여 사용
      
    3) DarkNet-53
      → DarkNet-19를 기반으로 Residual Network의 Shortcut Connection을 활용하고,
        크기를 늘려 53개의 Convolutional Layer를 가진 네트워크 구조 생성
      → DarkNet-19 보다는 무겁지만 ResNet-101 or 152보다는 가벼움
      
>**YOLO v3 실험 및 결과**

    1) Anchor Box x, y Offset Predictions
      → Linear Activation을 활용해 다양한 Box의 Width, Height의 x, y offset을
        기존의 Anchor Box 예측 방식을 사용해서 학습해보았지만
        오히려 모델의 안정성을 떨어뜨리고 성능이 좋지 않음.
      
    2) Linear x, y Prediction instead of Logistic
      → Logistic 함수 대신 선형 함수로 예측 결과 성능이 좋지 않음.
      
    3) Focal Loss
      → RetinalNet처럼 Class에 "배경"이라는 항목을 추가한 후 Focal Loss를 주는 방식을 사용할 수 있음.
        (Focal Loss : 배경은 Object에 비해 매우 빈도가 높아 모델이 배경에 대해서만 잘 학습하는 것을
                      방지하기 위해 Hard Sample과 Easy Sample의 Loss를 다르게 주는 것)
      
      → 하지만 YOLO 에서는 Object Score를 따로 계산하기 때문에 큰 의미가 없을 확률이 큼.
      
    4) Dual IOU and Truth Assignment
      → Faster R-CNN과 같이 0.7 이상의 IOU를 Positive Example, 0.3 이하를 Negative Example로
        취급하는 방식을 사용했지만 성능이 좋지 않음.
