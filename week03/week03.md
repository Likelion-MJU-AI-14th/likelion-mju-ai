# 3주차 과제: Tiny Shakespeare 데이터로더 및 입력 임베딩 구현

이번주 과제는 교재 Chapter2를 읽고 과제에서 제시하는 데이터셋에 대한 입력 임베딩을 파이썬 코드로 구현하는 것입니다.

교재에서 언급하는 토큰, 토크나이저, 임베딩, 데이터로더에 대한 개념을 익히고 토큰 임베딩과 위치 임베딩이 결합된 최종 입력 임베딩 행렬을 실제 코드로 생성하는 것을 목표로 합니다.

Chapter2를 읽고 코드를 참조하면 간단하게 수행할 수 있는 과제입니다

## 1. 환경 설정 및 데이터 준비
먼저 필요한 라이브러리를 임포트하고 오픈 소스 데이터셋인 Tiny Shakespeare 데이터를 불러옵니다. 토크나이저는 교재와 똑같이 `tiktoken` 라이브러리를 사용합니다.
```
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

torch.manual_seed('본인_학번')

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

## 2. 과제 구현
Chapter2 코드를 참고해서 과제에서 제시하는 요구사항을 .py 혹은 .ipynb(주피터노트북) 파일에 구현합니다. 

하이퍼파라미터는 아래와 같이 설정해주시면 감사하겠습니다.
+ batch_size = 1
+ output_dim = 8
+ context_length = 4
+ stride = 4

> **Note**
> 작성하신 코드에는 아래처럼 입력 임베딩에 대한 출력 코드가 포함되어야 합니다!

```
print("### 첫 번째 배치 입력 임베딩 결과 ###")
print(f"입력 텐서 형태 (Batch, Context, Embedding): {input_embeddings.shape}")
print("\n첫 번째 배치의 입력 임베딩 값:")
print(input_embeddings)
```

## 3. 과제 제출
### 3.1. 제출 위치
본인 이름 폴더 안에 `week03.py` 파일로 과제물을 제출해주세요.

```text
week03/{본인 이름}/week03.py
```


### 3.2. 제출 방법
코드를 작성한 파일을 기존과 똑같이 Pull Request 보내주시면 됩니다. 풀 리퀘 제목은 아래와 같습니다.

```text
[3주차 과제] 홍길동 과제 제출
```