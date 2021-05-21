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
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/2.SPPnet/4.PNG" width="800px" height="300px"></img><br/>  

## Fast R-CNN
>**Fast R-CNN Algorithms**

    -. R-CNN & SPPnet 대비 성능 개선 (학습시간 : 9배 / 3배,  실적용 : 213배 / 10배)
    -. ROI Pooling Layer를 통해 고정된 Feature Vector 생성
    -. Single-Stage로 학습 진행
    -. 전체 네트워크 업데이트 가능
    -. 저장 공간 불필요

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/1.PNG" width="800px" height="300px"></img><br/>  

    * ROI Pooling Layer : ROI 영역에 해당하는 부분만 Max Pooling을 통해 Feature Map으로부터 고정된 길이의 저차원 벡터로 축소
        - 각각의 ROI 는 (r, c, h, w)의 튜플 형태 [(r, c) : Top-Left Corner의 좌표]
        - h*w ROI 사이즈를 H*W의 작은 윈도우 사이즈로 나눔 (h/H * w/W)
        - SPPnet의 SPP Layer의 한 Pyramid Level 만 사용하는 형태는 동일
        
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/2.png" width="800px" height="300px"></img><br/>  
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

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/3.Fast_R-CNN/4.png" width="800px" height="100px"></img><br/>  

>**Fast R-CNN 한계**
    
    -. Region Proposal 단계가 대부분의 시간을 차지하기 때문에 Testing 단계에서 여전히 느림.

## Faster R-CNN
>**Faster R-CNN Algorithms**

    -. Region Proposal에 신경망을 적용[RPN(Region Proposal Network)]하여 시간 단축 (CPU → GPU)
    -. Region-Based 감지기의 Feature Map을 Region-Proposal Generating 에도 사용
        - 전체 Image를 Convolutional Network 하여 만들어낸 Feature Map을 Region-Proposal Generating 에서 활용
    -. RPN은 End to End로 학습 가능
    -. Object여부와 Bounding Box를 Regress하는 하나의 FCN

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/1.PNG" width="800px" height="500px"></img><br/>  

    * ROI Pooling Layer : ROI 영역에 해당하는 부분만 Max Pooling을 통해 Feature Map으로부터 고정된 길이의 저차원 벡터로 축소
        - 각각의 ROI 는 (r, c, h, w)의 튜플 형태 [(r, c) : Top-Left Corner의 좌표]
        - h*w ROI 사이즈를 H*W의 작은 윈도우 사이즈로 나눔 (h/H * w/W)
        - SPPnet의 SPP Layer의 한 Pyramid Level 만 사용하는 형태는 동일
        
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/2.PNG" width="800px" height="200px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/3.PNG" width="800px" height="300px"></img><br/>  

    * 학습 : 기본적으로 ImageNet을 사용한 Pre-Trained Network Base로 진행
    * 변경점
        - 마지막 Max Pooling Layer → ROI Pooling Layer로 대체(VGG16에서는 H=W=7)
        - 신경망의 마지막 FC Layer와 Softmax단이 두 개의 Output Layer로 대체(ImageNet : 1000개 분류)
        - 신경망의 입력이 Image와 ROI를 반영할 수 있도록 변경

>**Fast R-CNN vs Faster R-CNN**
    
    -. Region-wise Sampling → Hierarchical Sampling
        - SGD 미니배치가 Image에 종속됨(N개의 이미지를 각 R/N ROI개로)
        - N=2, R=128로도 좋은 학습 결과 확인
        - 약 64배 빠른 학습 가능(N 이 작을수록 계산복잡도 낮아짐)
    -. SPP Layer → ROI Pooling Layer
    -. Single-Stage(Multi-Task)
        - 최종 Classifier와 Regression까지 단방향 단계 [→ 효율적인 학습 (Softmax Classifier + Bounding Box Regressor)]

<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/3.PNG" width="800px" height="100px"></img><br/>  
----------------------------------------------------------------------------------------------------------------------------------------  
<img src="https://github.com/falling90/Object_Detection/blob/main/Reference/Image/4.Faster_R-CNN/4.PNG" width="800px" height="300px"></img><br/>  

>**Faster R-CNN 한계**
    
    -. Region Proposal 단계가 대부분의 시간을 차지하기 때문에 Testing 단계에서 여전히 느림.
