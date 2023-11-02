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


