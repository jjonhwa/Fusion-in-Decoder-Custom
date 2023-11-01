import random
import argparse
import os
import wandb

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch import nn
from transformers import (
    AutoTokenizer,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset


from tqdm import trange
from tqdm.auto import tqdm

from utils import (
    seed_everything, 
    deduplicate_question,
    mrr_calculate
)
from dataset import Cross_Dataset
from model import BertEncoder_For_CrossEncoder, RoBertaEncoder_For_CrossEncoder

def train(sub_args, c_encoder):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in c_encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in c_encoder.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        # eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    c_encoder.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    c_encoder.train()

    total_step = 0
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        loss_value = 0  # Use it when you use accumulation.
        losses = 0
        for step, batch in enumerate(epoch_iterator):
            batch = torch.transpose(batch, 0, 1)
            
            cross_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # "token_type_ids": batch[2]
            }

            for k in cross_inputs.keys():
                cross_inputs[k] = cross_inputs[k].tolist()

            # -- Make In-Batch Negative Sampling
            new_input_ids = []
            new_attention_mask = []
            if "roberta" not in sub_args.model:
                new_token_type_ids = [] 
            
            for i in range(len(cross_inputs["input_ids"])):
                sep_index = cross_inputs["input_ids"][i].index(tokenizer.sep_token_id)  # [SEP] token의 index

                for j in range(len(cross_inputs["input_ids"])):
                    
                    # -- Make Negative Samples => i_th query with j_th passage
                    # positive: i_th query + i_th passage
                    # negative: i_th query + j_th passage
                    # Note: Since multiple passages can be obtained for one query, the i_th query and j_th passage can be positive samples. 
                    #       Because of this, Sampling is performed in prepraration for this case. However, there is no significant difference in performance when shuffle is used as sampling
                    
                    query_id = cross_inputs["input_ids"][i][:sep_index]
                    query_att = cross_inputs["attention_mask"][i][:sep_index]
                    if "roberta" not in sub_args.model:
                        query_tok = cross_inputs['token_type_ids'][i][:sep_index]

                    context_id = cross_inputs["input_ids"][j][sep_index:]
                    context_att = cross_inputs["attention_mask"][j][sep_index:]
                    if "roberta" not in sub_args.model:
                        context_tok = cross_inputs['token_type_ids'][j][sep_index:] 
                    
                    query_id.extend(context_id)
                    query_att.extend(context_att)
                    if "roberta" not in sub_args.model:
                        query_tok.extend(context_tok) 
                    
                    new_input_ids.append(query_id)
                    new_attention_mask.append(query_att)
                    if "roberta" not in sub_args.model:
                        new_token_type_ids.append(query_tok) 

            new_input_ids = torch.tensor(new_input_ids)
            new_attention_mask = torch.tensor(new_attention_mask)
            if "roberta" not in sub_args.model:
                new_token_type_ids = torch.tensor(new_token_type_ids) 
            
            if torch.cuda.is_available():
                new_input_ids = new_input_ids.to("cuda")
                new_attention_mask = new_attention_mask.to("cuda")
                if "roberta" not in sub_args.model:
                    new_token_type_ids = new_token_type_ids.to('cuda') 

            if "roberta" not in sub_args.model:
                change_cross_inputs = {
                    "input_ids": new_input_ids,
                    "attention_mask": new_attention_mask,
                    'token_type_ids' : new_token_type_ids 
                }
            else:
                change_cross_inputs = {
                    "input_ids": new_input_ids,
                    "attention_mask": new_attention_mask,
                }

            cross_output = c_encoder(**change_cross_inputs) 
            cross_output = cross_output.view(-1, args.per_device_train_batch_size) # (batch_size, emb_dim)
            
            # only i_th element is accepted as positive
            targets = torch.arange(0, args.per_device_train_batch_size).long()

            if torch.cuda.is_available():
                targets = targets.to("cuda")

            score = F.log_softmax(cross_output, dim=1)
            loss = F.nll_loss(score, targets)

            ########################No ACCUMULATION#########################
            losses += loss.item()
            wandb.log({'train_loss': losses/(step+1)})

            if step % 100 == 0:
                print(f"Train {epoch}epoch loss: {losses/(step+1)}")

            c_encoder.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_step += 1

            # Validation Dataset에 대한 점수
            if total_step % 3000 == 0: # 3000
                with torch.no_grad():
                    c_encoder.eval()

                    result_scores = []
                    result_indices = []
                    for i in tqdm(range(len(valid_query))):
                        question = valid_query[i]
                        question_score = []

                        for passage in valid_document:
                            if "roberta" not in sub_args.model:
                                tokenized_examples = tokenizer(
                                    question,
                                    passage,
                                    truncation="only_second",
                                    max_length=512,
                                    return_token_type_ids=True,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                                    padding="max_length",
                                    return_tensors="pt",
                                )
                            else:
                                tokenized_examples = tokenizer(
                                    question,
                                    passage,
                                    truncation="only_second",
                                    max_length=512,
                                    return_token_type_ids=False,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                                    padding="max_length",
                                    return_tensors="pt",
                                )

                            score = 0
                            for j in range(len(tokenized_examples["input_ids"])):
                                input_ids = torch.tensor(tokenized_examples["input_ids"][j].unsqueeze(dim=0))
                                attention_mask = torch.tensor(tokenized_examples["attention_mask"][j].unsqueeze(dim=0))
                                if "roberta" not in sub_args.model:
                                    token_type_ids = torch.tensor(tokenized_examples["token_type_ids"][j].unsqueeze(dim=0))

                                if torch.cuda.is_available():
                                    input_ids = input_ids.to("cuda")
                                    attention_mask = attention_mask.to("cuda")
                                    if "roberta" not in sub_args.model:
                                        token_type_ids = token_type_ids.to("cuda")

                                if "roberta" not in sub_args.model:
                                    c_input = {
                                        "input_ids": input_ids,
                                        "attention_mask": attention_mask,
                                        "token_type_ids": token_type_ids,
                                    }
                                else:
                                    c_input = {
                                        "input_ids": input_ids,
                                        "attention_mask": attention_mask,
                                    }

                                tmp_score = c_encoder(**c_input)
                                if torch.cuda.is_available():
                                    tmp_score = tmp_score.to("cpu")
                                score += tmp_score

                            score = score / len(tokenized_examples["input_ids"])
                            question_score.append(score)

                        sort_result = torch.sort(torch.tensor(question_score), descending=True)
                        scores, index_list = sort_result[0], sort_result[1]

                        result_scores.append(scores.tolist())
                        result_indices.append(index_list.tolist())

                    valid_mrr = mrr_calculate(result_indices)
                    wandb.log({'Valid MRR': valid_mrr})
                    print('-' * 50)
                    print(f"{epoch} epoch - {step+1} steps valid mrr score:", valid_mrr)
                    print('-' * 50)

                backbone = sub_args.model.split("/")[1].split("-")[0]
                save_path = os.path.join(args.output_dir, f"cross_chunk_{backbone}_s{total_step+1}.pt")
                torch.save(c_encoder, save_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # --version
    parser.add_argument('--model', type=str, default='klue/roberta-large', help='You can insert "klue/bert-base" or "monologg/kobigbird-bert-base')
    parser.add_argument('--version', type=int, default=3, help='Choose 1 ~ 4')

    # -- training arguments
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate (default: 1e-5)")
    parser.add_argument('--train_batch_size', type=int, default=6, help="train batch size (default: 4)")
    parser.add_argument('--epochs', type=int, default=5, help="number of epochs to train (default: 10)")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="strength of weight decay (default: 0.01)")
    parser.add_argument('--warmup_steps', type=int, default=500, help="strength of weight decay (default: 500)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient accumulation steps (default: 1)")

    # --save
    parser.add_argument('--output_directory', type=str, default='./save/', help='Put in your save directory')
    parser.add_argument('--wandb_name', type=str, default='CrossEncoder - Chunk (512)')

    sub_args = parser.parse_args()

    args = TrainingArguments(
        output_dir=sub_args.output_directory,
        evaluation_strategy="epoch",
        learning_rate=sub_args.lr,
        # if you use bi-encoder, More batch size may be input than crossencoder.
        per_device_train_batch_size=sub_args.train_batch_size,
        gradient_accumulation_steps=sub_args.gradient_accumulation_steps,
        num_train_epochs=sub_args.epochs,
        weight_decay=sub_args.weight_decay,
    )

    # -- seed 고정
    seed_everything(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- dataset 불러오기 및 전처리
    train_data = load_dataset("jjonhwa/V3")

    document = train_data['train']['context']
    query = train_data['train']['question']

    total_idx = [i for i in range(len(document))]
    random.shuffle(total_idx)

    # train_length = round(len(total_idx) * 0.95)
    train_idx = total_idx[:-100] # 100
    valid_idx = total_idx[-100:]

    train_document = []
    train_query = []

    valid_document = []
    valid_query = []

    for idx in train_idx:
        train_document.append(document[idx])
        train_query.append(query[idx])

    for idx in valid_idx:
        valid_document.append(document[idx])
        valid_query.append(query[idx])

    train_query, train_document = deduplicate_question(train_query, train_document)
    valid_query, valid_document = deduplicate_question(valid_query, valid_document)
    print("Train Dataset의 개수:", len(train_query))
    print("Valid Dataset의 개수:", len(valid_query))

    # -- dataloader
    tokenizer = AutoTokenizer.from_pretrained(sub_args.model)
    
    train_dataset = Cross_Dataset(sub_args, train_query, train_document, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # -- model load
    if "roberta" in sub_args.model:
        c_encoder = RoBertaEncoder_For_CrossEncoder.from_pretrained(sub_args.model).to(device)
    elif "bert" in sub_args.model:
        c_encoder = BertEncoder_For_CrossEncoder.from_pretrained(sub_args.model).to(device)

    c_encoder = nn.DataParallel(c_encoder)

    wandb.init(project='Retrieval',
                group=None,
                entity=None,
                name=sub_args.wandb_name,
    )

    ## -- train
    train(sub_args, c_encoder)