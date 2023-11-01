# FiD-Custom
Unoffitial PyTorch implementation of [FiD-Light](https://arxiv.org/abs/2209.14290) and [FiDO](https://arxiv.org/abs/2212.08153) in Korean
또한, 코딩을 위해 다음의 페이지를 참고했다. [fid-official](https://github.com/facebookresearch/FiD) and [GQA](https://github.com/fkodom/grouped-query-attention-pytorch)

## Contents
FiD, FiD-Light, FiDO에 대하여 간략히 소개한다.

### FiD
<img width="1506" alt="image" src="https://github.com/jjonhwa/KLUE-NLI/assets/53552847/69ab9a83-9b69-426d-9e03-479ff1d6d3dc">
Retrieval을 통해 Top n Passage를 추출  ->  각각의 Q-P pair를 Encoder를 통과시켜 embedding vector 생성  ->  각 embedding vector를 Sequence 단위로 concatenate 수행  ->  Concatenated embedding vector를 Decoder에 통과시켜 정답 추출


### FiD-Light
<img width="1270" alt="image" src="https://github.com/jjonhwa/KLUE-NLI/assets/53552847/8d6a1e91-c309-49c4-a94f-1a1f8daa24a9">
- First-K: FiD에서 각 embedding vecotr를 Seqeunce 단위로 concatenate 수행할 때, 앞에서부터 K개의 Token에 대한 embedding만을 활용하여 concatenate을 수행한다. **(적용 O)**
- Source Pointing: 정답을 추출한 Evidence Passage의 Index를 함께 반환 -> 이를 활용하여 Passage를 Re-rank 수행하고, Re-ranked Passage를 활용하여 정답 추출 ( 적용 X )

### FiDO
- LSA(Layer-sparse cross-attention): (n, 2n, 3n, ...)번째 Layer에서의 Cross-Attention 만을 적용 **(적용 O)**
- MQA(Multi-query attention):  attention 수행 시 Multi Head에서 Key, Value는 single head로 share하여 적용. **(GQA로 적용)**
- Decoder Scaling: Decoder Model의 Size를 Scale Up. **(적용 O)**

## Data

### retrieval
- Retrieval 학습을 위한 코드

### preprocess
- FiD 학습을 위한 Retrieved Passage를 가지는 Dataset 생성

### fid
- FiD-custom 학습

## Run

### setup
```
bash requirements.sh
```

### retrieved dataset
if you want to make it directly, then use this code
```
python3 preprocess/fid_data.py
```

else, you can use `jjonhwa/SECOND_KQ_V2` dataset. It is linked with `fid` train code.

### fid
- FiDT5: can apply First_K, LSA, GQA ( 성능 비교한 후, 적용할만한 기법을 탐색 실험 수행 )
- FiDSKT: can apply First_K, LSA ( LSA의 성능이 좋았기 때문에, Decoder Scaling을 수행한 후에 GQA를 적용하지 않음)

```
python3 fid/FiDT5_train.py --n_cross_layer 6 --batch_size 4 --wandb_name {your_name}
```


```
# large / per batch 2 /
# Original
python3 FiDT5_train.py --wandb_name "FiD-Original"

# First_K
python3 FiDT5_train.py --first_k 8 --wandb_name "FiD-K8"
python3 FiDT5_train.py --first_k 32 --wandb_name "FiD-K32"

# GQA
python3 FiDT5_train.py --kv_heads 4 --wandb_name "FiD-GQA4"

# LSA
python3 FiDT5_train.py --n_cross_layer 6 --wandb_name "FiD-LSA6"
python3 FiDT5_train.py --n_cross_layer 4 --wandb_name "FiD-LSA4"


# LSA & GQA
python3 FiDT5_train.py --kv_heads 4 --n_cross_layer 6 --wandb_name "FiD-LSA6-GQA4"

######### First_K 조합 시 8로 고정 ###############################################
# First_K & GQA
python3 FiDT5_train.py --first_k 8 --kv_heads 4 --wandb_name "FiD-K8-GQA4"
# python3 FiDT5_train.py --first_k 32 --kv_heads 4 --wandb_name "FiD-K32"

# LSA & First_K
python3 FiDT5_train.py --first_k 8 --n_cross_layer 6 --wandb_name "FiD-K8-LSA6"

# LSA & First_K & GQA
python3 FiDT5_train.py --first_k 8 --n_cross_layer 6 --kv_heads 4 --wandb_name "FiD-K8-LSA6-GQA4"


# FiDT5-Summary
python3 FiDT5_train.py --n_cross_layer 6 --wandb_name "FiDT5-SUMMARY(linear)_V2-n10-e5(example)" --data 'jjonhwa/SECOND_KOWIKI_RETRIEVE_200_V2' --n_context 10 --epochs 5

# FiDSKT-Summary
python3 FiDSKT_train.py --n_cross_layer 6 --wandb_name "FiDSKT-SUMMARY(CyclicLR)_V2-n5-e10(example)" --data 'jjonhwa/SECOND_KOWIKI_RETRIEVE_300_V2' --n_context 5 --epochs 10
```