# FiD-Custom
- Unoffitial PyTorch implementation of [FiD](https://arxiv.org/pdf/2007.01282.pdf), [FiD-Light](https://arxiv.org/abs/2209.14290) and [FiDO](https://arxiv.org/abs/2212.08153) in Korean
- 또한, 코딩을 위해 다음의 페이지를 참고했다. [fid-official](https://github.com/facebookresearch/FiD) and [GQA](https://github.com/fkodom/grouped-query-attention-pytorch)

## Contents

### FiD
<img width="1506" alt="image" src="https://github.com/jjonhwa/KLUE-NLI/assets/53552847/69ab9a83-9b69-426d-9e03-479ff1d6d3dc">

### FiD-Light
<img width="1270" alt="image" src="https://github.com/jjonhwa/KLUE-NLI/assets/53552847/8d6a1e91-c309-49c4-a94f-1a1f8daa24a9">
- First-K: FiD에서 각 embedding vecotr를 Seqeunce 단위로 concatenate 수행할 때, 앞에서부터 K개의 Token에 대한 embedding만을 활용하여 concatenate을 수행한다. **(적용 O)**
- Source Pointing: 정답을 추출한 Evidence Passage의 Index를 함께 반환 -> 이를 활용하여 Passage를 Re-rank 수행하고, Re-ranked Passage를 활용하여 정답 추출 ( 적용 X )

### FiDO
(그림 삽입)
- LSA(Layer-sparse cross-attention): (n, 2n, 3n, ...)번째 Layer에서의 Cross-Attention 만을 적용 **(적용 O)**
- MQA(Multi-query attention):  attention 수행 시 Multi Head에서 Key, Value는 single head로 share하여 적용. **(GQA로 적용)**
- Decoder Scaling: Decoder Model의 Size를 Scale Up. **(적용 O)**

### Retrieval
- if you want to train your own retrieval model. then, check [this repository](https://github.com/jjonhwa/Cross-Encoder-with-Bi-Encoder)
- 위 레포지토리에 기반하여 code를 작성하였다.

## Data
```
+- fid 
    |   +- src 
    |   +- FiDT5_train.py
    |   +- FiDSKT_train.py

+- preprocess (make retrieved dataset)
    |   +- fid_data.py
    |   +- preprocess.py

+- retrieval (train retrieval model)
    |   +- datatset.py
    |   +- model.py
    |   +- retreiver_train.py
    |   +- utils.py

+- inference.py
+- requirements.sh
```

## Run

### setup
```
bash requirements.sh
```

### preprocess
- make retrieved dataset (If you have a lot of data, then, 굉장히 오랜시간이 걸린다.)
- you can use already made `jjonhwa/SECOND_KQ_V2` dataset. It is linked with `fid` train code.
(이미 만들어진 데이터를 활용할 수 있다.)

```
python3 preprocess/fid_data.py
```

### fid

```
# Original fid
python3 fid/FiDT5_train.py

# with first_k
# python3 fid/FiDT5_train.py --first_k 8

# with LSA
# python3 fid/FiDT5_train.py --n_cross_layer 6

# with GQA
# python3 fid/FiDT5_train.py --kv_heads 4

# 위의 것들을 조합해서 활용할 수 있다. 다음과 같이
# python3 fid/FiDT5_train.py --n_cross_layer 6 --kv_heads 4 --first_k 8

# with decoder scaling
# python3 fid/FiDSKT_train.py

# with decoder scaling & LSA
# python3 fid/FiDSKT_train.py --n_cross_layer 6
```

## Experiments
<img width="784" alt="image" src="https://github.com/jjonhwa/FiD-Custom/assets/53552847/5deec3d7-404b-48fd-951e-48d4e3bf01ca">
<img width="784" alt="Screen Shot 2023-11-02 at 5 42 30 PM" src="https://github.com/jjonhwa/AGC_rev/assets/53552847/2e9255fa-bd4b-4768-8f0f-6eabb862d524">
<img width="784" alt="Screen Shot 2023-11-02 at 5 41 08 PM" src="https://github.com/jjonhwa/AGC_rev/assets/53552847/32188034-658e-49ec-acc2-f4ddb840b794">
<img width="784" alt="Screen Shot 2023-11-02 at 5 40 40 PM" src="https://github.com/jjonhwa/AGC_rev/assets/53552847/a6dc39e1-a5f9-49fd-984c-67a665224551">

|                          | EVAL EM | EVAL TIME | MODEL PARAMETERS |
| ------------------------ | ------- | --------- | ---------------- |
| FiDT5 Original           | 39.63   | 1,519s    | 783,019,008      |
| FiDT5 K8 (FiD-Light)     | 26.37   | 1,215s    | 783,019,008      |
| FiDT5 K32 (FiD-Light)    | 26.33   | 1,233s    | 783,019,008      |
| FiDT5 LSA6 (FiDO)        | **37.12**   | **1,268s**    | **699,112,448**      |
| FiDT5 LSA4 (FiDO)        | 36.68   | 1,265s    | 707,503,104      |
| FiDT5 GQA4 (FiDO)        | 23.00   | 1,271s    | 783,019,008      |
| FiDSKT Original (FiDO)   | .       | .         | 1,892,694,144    |
| FiDSKT LSA6 (FiDO)       | 10.82   | 2,576s    | 1,597,551,744    |

- EVAL EM: 가장 좋은 성능
- EVAL TIME: 가장 좋은 성능을 낸 순서대로 3개의 Step에서의 evaluation time의 평균

### Analysis
- First-K, LSA, GQA 모두 evaluation time 측면에서 개선
- 하지만, 성능의 희생을 많이 감수해야 함.
- LSA의 경우, evaluation time은 줄이면서도 성능 하락의 폭이 그리 크지 않음.
- LSA에서 4와 6의 차이가 크지 않은 것으로 보아, Cross Attention 개수의 감소가 일정 수준에서 
- FiDSKT의 경우, Cross Attention의 파라미터가 학습되지 않아서 성능이 낮은 것으로 보인다. 추가적으로 학습을 진행할 경우 성능 개선의 여지를 기대한다.