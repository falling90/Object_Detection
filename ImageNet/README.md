# ImageNet

>**ImageNet이란?**

    -. WordNet 계층 구조에 따라 구성된 이미지 데이터베이스로 계층 구조의 각 노드가 수백, 수천 개의 이미지로 표시됨.
    -. 컴퓨터 비전과 딥 러닝 연구를 발전시키는 데 중요한 역할을 했고, 비상업적 용도로 데이터는 무료로 제공됨.

>**History**

    -. AlexNet(2012) → ZFNet(2013) → VGGNet(2015) → ResNet(2015)
    
    1) AlexNet(2012) : 심층 합성곱 신경망을 사용한 ImageNet 분류
    2) ZFNet(2013) : 합성곱 신경망의 시각화
    3) VGGNet(2015) : 대규모 이미지(Large scale) 인식을 위한 초 심층 합성곱 신경망
    4) ResNet(2015) : 이미지 인식을 위한 Deep Residual Learning(심층 잔차 학습)
<img src="https://github.com/falling90/Object_Detection/blob/main/ImageNet/Image/1.PNG" width="1000px" height="400px"></img><br/>   

## AlexNet(2012)
>**구조**

    -. LeNet(최초의 CNN, 1988)과 전반적인 구조는 유사함.
    -. 단, 2개의 GPU로 병렬연산을 수행하기위해 병렬적인 구조로 설계됨 [당시 H/W 한계로 인하여 이와 같이 구성됨(GPU Memory 용량 부족)]
<img src="https://github.com/falling90/Object_Detection/blob/main/ImageNet/Image/2.PNG" width="1000px" height="400px"></img><br/>   

>**특징**

    -. Overlapped Max Pooling 방식 사용
    -. Activation Funcion : ReLU(Rectified Linear Unit)
    -. LRN(Local Response Normalization) 적용 (ReLU 함수의 단점을 보완하기 위해 적용)
<img src="https://github.com/falling90/Object_Detection/blob/main/ImageNet/Image/3.PNG" width="1000px" height="400px"></img><br/>   

>**Overfitting 방지**

    -. 작은 연산만으로 학습데이터를 늘리는 Data Augmentation 방법 적용
      1) Image 개수 증가 : 상하좌우, 반전 등
      2) RGB Color Channel 값 변경
    -. Drop-out 적용 : 학습시간 단축 & Overfitting 방지
<img src="https://github.com/falling90/Object_Detection/blob/main/ImageNet/Image/4.PNG" width="800px" height="400px"></img><br/>   
