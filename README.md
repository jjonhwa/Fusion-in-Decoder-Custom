# FiD-Custom
- (Unoffitial) PyTorch implementation of [FiD](https://arxiv.org/pdf/2007.01282.pdf), [FiD-Light](https://arxiv.org/abs/2209.14290) and [FiDO](https://arxiv.org/abs/2212.08153)
- Also, the following pages were referenced for coding. [fid-official](https://github.com/facebookresearch/FiD) and [GQA](https://github.com/fkodom/grouped-query-attention-pytorch)

## Contents

### FiD
<img width="1506" alt="image" src="https://github.com/jjonhwa/KLUE-NLI/assets/53552847/69ab9a83-9b69-426d-9e03-479ff1d6d3dc">

### FiD-Light
<img width="1270" alt="image" src="https://github.com/jjonhwa/KLUE-NLI/assets/53552847/8d6a1e91-c309-49c4-a94f-1a1f8daa24a9">

- First-K: When concatenate each embedding vector to single sequence in FiD, concatenation is performed using only the incorporation of K tokens from the beginning. **(Applied)**
- Source Pointing: Return the index of the evidence passages from which the correct answer was extracted -> Use it to perform Re-rank on the passages, and use re-ranked passages to extract the correct answer. **(Not Applied)**

### FiDO
![Screen Shot 2023-11-06 at 11 03 00 AM](https://github.com/jjonhwa/FiD-Custom/assets/53552847/18751e68-a537-45bd-a94a-0272553d3e3b)
![Screen Shot 2023-11-06 at 11 03 56 AM](https://github.com/jjonhwa/FiD-Custom/assets/53552847/8fc71a84-956a-4dea-aede-7e1b187537a4)

- LSA(Layer-Sparse cross-Attention): Just apply the cross-attention at (n, 2n, 3n, ..)th layer. **(Applied)**
- MQA(Multi-Query Attention): When applying MHA, Key and Value are applied by sharing one single head. **(Applied as GQA)**
- Decoder Scaling: Scale up the decoder model's size. **(Applied)**
- **NOTE:** It was implemented only by increasing the size with a decoder model based on GPT-2. If you want to connect other Large models, please refer to the `fid/FiDSKT_train.py`. In addition, in the case of decoder scaling, only First-K and LSA are applicable.

### Retrieval
- If you want to train your own retrieval model. then, check [this repository](https://github.com/jjonhwa/Cross-Encoder-with-Bi-Encoder)
- The code related to the retrieval was created based on the above repository

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
- Make retrieved dataset (If you have a lot of data, then, It takes a very long time. If someone can optimize that code, please pull-request to me)
- Can use already made `jjonhwa/SECOND_KQ_V2` dataset. It is linked with `fid` train code.

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

# The above can be combined and utilized. as follows
# python3 fid/FiDT5_train.py --n_cross_layer 6 --kv_heads 4 --first_k 8

# with decoder scaling
# python3 fid/FiDSKT_train.py

# with decoder scaling & LSA
# python3 fid/FiDSKT_train.py --n_cross_layer 6
```

### inference
- If you want to experience `inference.py`, then need to download the fid-trained model. (example model is fine-tuned by summarization dataset)
- But, that model is not well trained. It is trained with KorQuAD Dataset and slightly overfitted that data.

```
# download fid-trained model
gdown https://drive.google.com/uc?id=1bg8tCSImGuNQGNrGdR99RcY_pFHZJUT0

# Go inference
python3 inference.py
```



## Experiments

### FiDT5 Original
<img width="784" alt="Screen Shot 2023-11-06 at 2 16 33 PM" src="https://github.com/jjonhwa/FiD-Custom/assets/53552847/49724e17-a0a9-4877-a658-0ae028f5642c">

### FiDT5 Original, LSA and FiDSKT
<img width="784" alt="Screen Shot 2023-11-02 at 5 42 30 PM" src="https://github.com/jjonhwa/AGC_rev/assets/53552847/2e9255fa-bd4b-4768-8f0f-6eabb862d524">

### FiDT5 LSA and GQA
<img width="784" alt="Screen Shot 2023-11-02 at 5 41 08 PM" src="https://github.com/jjonhwa/AGC_rev/assets/53552847/32188034-658e-49ec-acc2-f4ddb840b794">

### FiDT5 First-K
<img width="784" alt="Screen Shot 2023-11-02 at 5 40 40 PM" src="https://github.com/jjonhwa/AGC_rev/assets/53552847/a6dc39e1-a5f9-49fd-984c-67a665224551">

### Model Info
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

- EVAL EM: Best Evaluation Score
- EVAL TIME: Average evaluation times in the 3 steps in order of best performance
- Backbone Model(FiDT5): `KETI-AIR/ke-t5-large`
- Backbone Model(FiDSKT): `Encoder -> KETI-AIR/ke-t5-large(encoder)`, `Decoder -> skt/ko-gpt-trinity-1.2B-v0.5`
- n_context: `FiDT5 - 10`, `FiDSKT - 5`
- Learning Rate: `1e-4`
- Optimizer: `Adam`
- batch_size: `2 per GPU`



### Analysis
- All methods improve evaluation time, but degrade performance
- But, **LSA don't downgrade performance much** and **also get great improve evaluation time**
- The difference between LSA4 and LSA6 is not significant. based on this point, difference in the number of cross-attention does not show a significant difference in performance when reduced more than a certain number.
- In the case of FiDSKT, the performance seems to be low because the parameters of Cross-Attention are not learned. If additional learning is conducted, there is room for performance improvement.

