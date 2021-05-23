# Object Detection Algorithms

>**발전 과정**

    -. R-CNN → SPPnet → Fast R-CNN → Faster R-CNN → YOLO(v1 to v5)   
   

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

## OverView
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/9.png" width="1000px" height="250px"></img><br/>  
