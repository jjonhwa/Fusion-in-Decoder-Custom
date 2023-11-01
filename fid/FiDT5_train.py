import time
import argparse
import random
import wandb
import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm

import torch
import transformers

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, Dataset, load_metric
from torch import nn

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.options import Options
from src.modeling_t5 import T5ForConditionalGeneration

metric_squad = load_metric('squad')
metric_bleu = load_metric('bleu')
metric_rouge = load_metric('rouge')

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset) # Random 사용 안하니까
    dataloader = DataLoader(
        dataset,
        sampler = sampler,
        batch_size = opt.per_gpu_batch_size * 16,  # evaluation에서만 늘리기 
        drop_last = False,
        collate_fn = collator
    )
    model.eval()

    model = model.module if hasattr(model, 'module') else model

    generated_examples = []

    ans_list = []
    gold_list = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids = context_ids.cuda(),
                attention_mask = context_mask.cuda(),
                max_length = 1024, # Summarization: 1024
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                generated_examples.append(ans)

                gold_from = dataset.get_example(idx[k])
                
                if 'answers' in gold_from:
                    gold = gold_from['answers']
                else: # Summary
                    gold = [gold_from['summary']]

                ans_list.append(ans)
                gold_list.append(gold)

    total_score = src.evaluation.get_score(
        ans_list, 
        gold_list,
        tokenizer,
        metric_bleu,
        metric_rouge,
        metric_squad,
        opt,
    )
    generated_index = random.sample(range(len(generated_examples)), 5)
    new_df = pd.DataFrame({"Random1": [generated_examples[generated_index[0]]],
                           "Random2": [generated_examples[generated_index[1]]],
                           "Random3": [generated_examples[generated_index[2]]],
                           "Random4": [generated_examples[generated_index[3]]],
                           "Random5": [generated_examples[generated_index[4]]],})
    return total_score, new_df
    

def train(model, train_dataloader, opt):
    # opt.lr = 1e-3 # FiD: 1e-4, FiD-Light: 1e-3
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    loss, curr_loss = 0.0, 0.0
    dev_em, best_em = 0.0, -1.0

    epoch = 0
    model.train()

    df = pd.DataFrame({"Random1": [],
                       "Random2": [],
                       "Random3": [],
                       "Random4": [],
                       "Random5": [],})
    # while step < opt.total_steps:
    while True:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader)):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids = context_ids.cuda(),
                attention_mask = context_mask.cuda(),
                labels = labels.cuda(),
                return_dict = False,
            )[0]

            # DataParallel을 수행할 때, Loss가 GPU 수 만큼 반환하게 되어, Vector를 출력하게 됨.
            # 그래서 Backward를 수행할 떄는 Scalar가 반환되어야 하는데 Vector라서 Error가 발생
            # 즉, mean을 취한 후에 backward를 해야지 에러가 발생하지 않음.
            train_loss.mean().backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.mean().item()

            wandb.log({'train_loss': train_loss.mean()})
            if step % opt.eval_freq == 0:

                start_time = time.time()
                dev_score, new_df = evaluate(model, eval_dataset, tokenizer, collator, opt)
                dev_em, dev_f1, dev_bleu, dev_r1, dev_r2, dev_rL, dev_rLsum = dev_score
                end_time = time.time()

                if dev_em > best_em:
                    torch.save(model.state_dict(), f'/home/ubuntu/AGC_second/FiD/save/SUMMARY_FiDT5_best_s{step}_V2.pt')
                    best_em = dev_em

                wandb.log({'eval_em': dev_em*100})
                # wandb.log({'eval_squad_em': dev_squad_em})
                wandb.log({'eval_f1': dev_f1})
                wandb.log({'eval_bleu': dev_bleu*100})
                wandb.log({'eval_r1': dev_r1*100})
                wandb.log({'eval_r2': dev_r2*100})
                wandb.log({'eval_rL': dev_rL*100})
                wandb.log({'eval_rLsum': dev_rLsum*100})
                wandb.log({'eval_time': end_time - start_time})

                df = pd.concat([df, new_df], axis=0)
                wandb.log({"Generated Examples": wandb.Table(dataframe=df)})

                # save code 삭제
                print("-" * 50)
                print(f"{step} / {opt.total_steps} |")
                print(f"train: {curr_loss / opt.eval_freq:.3f} |")
                print(f"evaluation: {100 * dev_em: .2f}EM | {dev_f1: .2f}F1(squad) | {100 * dev_bleu: .2f}BLEU | {100 * dev_r1: .2f}R1 | {100 * dev_r2: .2f}R2 | {100 * dev_rL: .2f}RL | {100 * dev_rLsum: .2f}RLsum")
                print(f"lr: {scheduler.get_last_lr()[0]:.5f}")
                print(f"10 passages evaluation time: {end_time - start_time:.4f}")
                
                model.train()
                curr_loss = 0.
                
        # 1 Epoch에서 종료
        if epoch == opt.epochs:
            break

def pasasge_more_than(data: pd.DataFrame, num: int):
    """
        data:
            KorQuAD 2.0
            Retriever에서 Token 200으로 쪼개서 관련있는 Passage를 10개 받는다.
                그 과정에서 Document 길이가 짧을 경우 모두 반환해도 10개가 되지 않는 Doc이 존재
        num:
            FiD에서 활용할 Passage의 개수
    """    

    # Passage의 개수가 10개가 넘지 않은 데이터가 존재함 
    count_dict = defaultdict(int)
    for i in range(len(data)):
        count_dict[len(data['ctxs'][i])] += 1

    # Passage의 개수가 num개 이상인 데이터만 활용
    drop_index = []
    for i in range(len(data)):
        if len(data['ctxs'][i]) < num:
            drop_index.append(i)

    data.drop(drop_index, axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

def get_dataset(opt):

    # data = load_dataset("jjonhwa/SECOND_KQ_V2")
    data = load_dataset(opt.from_data)

    new_data = pd.DataFrame(data['train'])

    new_data = pasasge_more_than(new_data, opt.n_context)

    train_data = new_data.sample(round(len(new_data) * 0.95), random_state=42)

    eval_idx = [idx for idx in range(len(new_data)) if idx not in train_data.index]
    eval_data = new_data.iloc[eval_idx]

    train_data = Dataset.from_pandas(train_data)
    eval_data = Dataset.from_pandas(eval_data)

    train_data = [d for d in train_data]
    eval_data = [d for d in eval_data]

    # score 생성하는 부분 삭제 ( NQ에 Score가 없으므로 )
    train_examples = []
    for k, example in enumerate(train_data):
        if not 'id' in example:
            example['id'] = k
        
        train_examples.append(example)

    eval_examples = []
    for k, example in enumerate(eval_data):
        if not 'id' in example:
            example['id'] = k
        
        eval_examples.append(example)


    train_dataset = src.data.Custom_Dataset(train_examples, opt.n_context)
    eval_dataset = src.data.Custom_Dataset(eval_examples, opt.n_context)

    return train_dataset, eval_dataset

if __name__ == '__main__':

    # 주요 변경해야할 사항: lr, eval_freq, per_gpu_batch_size, n_context, model_name, first_k
    parser = argparse.ArgumentParser()

    # -- model ( FiD-LIGHT, LSA, GQA )
    parser.add_argument('--n_context', type=int, default=10, help='number of context to use for FiD')
    parser.add_argument('--kv_heads', type=int, default=-1, help='number of head about K and V')
    parser.add_argument('--first_k', type=int, default=-1, help='only use first_k tokens from encoder_hidden_states')
    parser.add_argument('--n_cross_layer', type=int, default=-1, help="'only use n_cross_layer'th block's EncDecAttention")
    
    # -- HyperParameter
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument("--eval_freq", type=int, default=1000) # eval_freq 1000으로 바꾸기

    # -- Dataset (MRC or Summarization)
    parser.add_argument('--data', type=str, default='jjonhwa/SECOND_KQ_V2', help='choose Retrieved Dataset \
                                                                                    ( MRC(jjonhwa/SECOND_KQ_V2) or \
                                                                                    SUMMARY(jjonhwa/SECOND_KOWIKI_RETRIEVE_{200 or 300}_V2 or \
                                                                                            jjonhwa/SECOND_RETRIEVE_PROCESSED_150)')

    # -- wandb
    parser.add_argument('--wandb_name', type=str, default='FiD')
    parser.add_argument("--wandb_project", type=str, default='Generation')
    sub_args = parser.parse_args()

    options = Options()
    options.add_reader_options() # reader hyperparameter setting
    options.add_optim_options() # Optimizer Setting
    opt = options.parse() # argparse
    
    opt.n_context = sub_args.n_context
    opt.kv_heads = sub_args.kv_heads
    opt.first_k = sub_args.first_k
    opt.n_cross_layer = sub_args.n_cross_layer
    
    opt.epochs = sub_args.epochs
    opt.per_gpu_batch_size = sub_args.batch_size
    opt.eval_freq = sub_args.eval_freq
    
    opt.from_data = sub_args.data
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(opt.seed)

    # slurm: Cluster Server 상에서 작업을 관리하기 위한 프로그램, Node간의 통신을 통해 작업 관리를 함.
    src.slurm.init_distributed_mode(opt) # GPU Handler
    src.slurm.init_signal_handler()

    # model_name = "KETI-AIR/ke-t5-large"
    tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_name)

    # collator에서 Batch별 Max Length로 수행하는 걸로 보임
    collator = src.data.Collator(opt.text_maxlength,
                                tokenizer,
                                answer_maxlength=opt.answer_maxlength)

    # Train: 90% / Eval: 10%
    train_dataset, eval_dataset = get_dataset(opt)

    # `device_map`을 주면 Loading 속도가 빨라짐. 왜 일까??
    t5 = T5ForConditionalGeneration.from_pretrained(opt.model_name, device_map="auto")
    model = src.model.FiDT5(t5.config, first_k=opt.first_k if opt.first_k != -1 else None) # config를 init으로 받아야 함
    model.load_t5(t5.state_dict())
    model = model.to(opt.device)

    # Without GQA & LSA ( Original FiD )
    if opt.kv_heads == -1 and opt.n_cross_layer == -1: # w/o GQA, LSA
        print("Original")
        pass
    
    #  With LSA ( FiD + LSA )
    elif opt.kv_heads == -1:
        print("with LSA")
        model = src.model.convert_LSA(model, n_cross_layer=opt.n_cross_layer)
    
    # With GQA ( FiD + GQA )
    elif opt.n_cross_layer == -1:
        print("with GQA")
        model = src.model.convert_GQA(model, kv_heads=opt.kv_heads)
    
    # With GQA & LSA
    else: 
        print("with LSA & GQA")
        model = src.model.convert_GQA(model,kv_heads=opt.kv_heads)
        model = src.model.convert_LSA(model,n_cross_layer=opt.n_cross_layer)
    
    model = nn.DataParallel(model)

    # 1은 opt.global_rank를 대체한 값이다
    torch.manual_seed(opt.seed + opt.global_rank) # global rank에 따라서 train 시에 다른 seed를 줌 ( 왜 인지는 모르겠음 )

    train_sampler = RandomSampler(train_dataset)

    # get_item ==> index, target_ids, target_mask, passage_ids, passage_masks
    train_dataloader = DataLoader(
        train_dataset,
        sampler = train_sampler,
        batch_size = opt.per_gpu_batch_size,
        drop_last = True,
        collate_fn = collator
    )

    # opt.eval_freq = 3000
    wandb.init(project=sub_args.wandb_project,
                group=None,
                entity=None,
                name=sub_args.wandb_name,
    )

    opt.total_steps = len(train_dataloader) * opt.epochs
    print(opt)
    
    train(model, train_dataloader, opt)

