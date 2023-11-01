import os
import argparse
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import preprocess

def get_topn_passage_QA(
        model,
        tokenizer,
        data: pd.DataFrame,
        n: int = 10,
        passage_length: int = 200,
):
    
    new_data = []

    with torch.no_grad():
        model.eval()

        contexts = data['context']
        questions = data['question']
        answers = data['answer']
        
        for context, question, answer in tqdm(zip(contexts, questions, answers)):

            tokenized_context = tokenizer(
                question,
                context,
                truncation='only_second',
                max_length=passage_length,
                stride=int(passage_length * 0.2),
                return_overflowing_tokens=True,
                padding='max_length',
                return_tensors='pt'
            )

            output_true, output_false = [], []
            result_scores, result_indices = [], []

            overflowing_count = len(tokenized_context['input_ids'])
            for i in range(overflowing_count):

                input_ids = tokenized_context['input_ids'][i].unsqueeze(dim=0).to(device)
                attention_mask = tokenized_context['attention_mask'][i].unsqueeze(dim=0).to(device)
                token_type_ids = tokenized_context['token_type_ids'][i].unsqueeze(dim=0).to(device)

                model_input = {"input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids}
                
                tmp_score = model(**model_input)
                logits = tmp_score[0]
                prob = F.softmax(logits, dim=-1).detach().cpu().numpy() 
                logits = logits.detach().cpu().numpy()
                result = np.argmax(logits, axis=-1) # 0 or 1 ( Binary Prediction )
                
                if result.item() == 1:
                    output_true.append((np.max(prob), i))
                    output_true.sort(key = lambda x: -x[0])
                else:
                    output_false.append((np.max(prob), i))
                    output_false.sort(key = lambda x: x[0])

            for i in range(len(output_true)):
                if len(result_scores) < n:
                    result_scores.append((output_true[i][0], "positive"))
                    result_indices.append(output_true[i][1])
            for i in range(len(output_false)):
                if len(result_scores) < n:
                    result_scores.append((output_false[i][0], "negative"))
                    result_indices.append(output_false[i][1])

            candidate_text = []
            for i, idx in enumerate(result_indices):
                decoded_text = retriever_tokenizer.decode(tokenized_context['input_ids'][idx])
                decoded_text = decoded_text.split("[SEP]")
                text = decoded_text[1].strip()
                candidate_text.append((text, result_scores[i]))

            ctxs = []
            for text, score_pair in candidate_text:
                
                score, bool_value = score_pair
                if bool_value == "negative":
                    score *= -1

                ctxs_dict = {"text": text, "score": score}
                ctxs.append(ctxs_dict)

            each_data = {"question": question,
                        "answers": [answer],
                        "ctxs": ctxs}
            
            new_data.append(each_data)

    return new_data

def get_topn_passage_SUMMARY(
        model,
        tokenizer,
        data: pd.DataFrame,
        n: int = 10,
        passage_length: int = 200,
):
    
    new_data = []

    with torch.no_grad():
        model.eval()

        contexts = data['context']
        summaries = data['summary']
        
        base_length = 512 - passage_length
        temp_summary = [tokenizer.decode(tokenizer.encode(summaries[i])[:base_length], skip_special_tokens=True) for i in range(len(summaries))]
        summary_token_length = [len(tokenizer.encode(s)) for s in temp_summary]
        
        for idx, (context, summary) in tqdm(enumerate(zip(contexts, summaries))):
            
            temp_s = temp_summary[idx]
            temp_s_length = summary_token_length[idx]

            tokenized_context = tokenizer(
                temp_s,
                context,
                truncation='only_second',
                max_length=passage_length + temp_s_length,
                stride=int((passage_length + temp_s_length) * 0.2),
                return_overflowing_tokens=True,
                padding='max_length',
                return_tensors='pt'
            )

            output_true, output_false = [], []
            result_scores, result_indices = [], []

            overflowing_count = len(tokenized_context['input_ids'])
            for i in range(overflowing_count):

                input_ids = tokenized_context['input_ids'][i].unsqueeze(dim=0).to(device)
                attention_mask = tokenized_context['attention_mask'][i].unsqueeze(dim=0).to(device)
                token_type_ids = tokenized_context['token_type_ids'][i].unsqueeze(dim=0).to(device)

                model_input = {"input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids}
                
                tmp_score = model(**model_input)
                logits = tmp_score[0]
                prob = F.softmax(logits, dim=-1).detach().cpu().numpy() 
                logits = logits.detach().cpu().numpy()
                result = np.argmax(logits, axis=-1) # 0 or 1 ( Binary Prediction )
                
                if result.item() == 1:
                    output_true.append((np.max(prob), i))
                    output_true.sort(key = lambda x: -x[0])
                else:
                    output_false.append((np.max(prob), i))
                    output_false.sort(key = lambda x: x[0])

            for i in range(len(output_true)):
                if len(result_scores) < n:
                    result_scores.append((output_true[i][0], "positive"))
                    result_indices.append(output_true[i][1])
            for i in range(len(output_false)):
                if len(result_scores) < n:
                    result_scores.append((output_false[i][0], "negative"))
                    result_indices.append(output_false[i][1])

            candidate_text = []
            for i, idx in enumerate(result_indices):
                decoded_text = retriever_tokenizer.decode(tokenized_context['input_ids'][idx])
                decoded_text = decoded_text.split("[SEP]")
                text = decoded_text[1].strip()
                candidate_text.append((text, result_scores[i]))

            ctxs = []
            for text, score_pair in candidate_text:
                
                score, bool_value = score_pair
                if bool_value == "negative":
                    score *= -1

                ctxs_dict = {"text": text, "score": score}
                ctxs.append(ctxs_dict)

            each_data = {'ctxs': ctxs, 'summary': summary}
            
            new_data.append(each_data)

    return new_data

def preprocessing(context):
    context = [context]
    context = preprocess.remove_email(context)
    context = preprocess.remove_user_mention(context)
    context = preprocess.remove_url(context)
    context = preprocess.remove_bad_char(context)
    context = preprocess.remove_press(context)
    context = preprocess.remove_copyright(context)
    context = preprocess.remove_photo_info(context)
    context = preprocess.remove_useless_bracket(context)
    context = preprocess.remove_repeat_char(context)
    context = preprocess.clean_punc(context)
    context = preprocess.remove_repeated_spacing(context)
    
    if context is None:
        return ""
    return context[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_context", type=int, default=10)
    parser.add_argument("--passage_length", type=int, default=200)
    
    parser.add_argument("--save_folder", type=str, default='./data')
    parser.add_argument("--save_file", type=str, default='SECOND_KQ_V2.pkl')
    parser.add_argument(
        "--from_data", 
        type=str, 
        default='jjonhwa/raw5_v1',
        help="The form when transformed into Pandas is as follows \
            | context | question | answer | answer_start | (But, don't use answer_start) or \
            | context | summary |"
    )

    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    retrieval = AutoModelForSequenceClassification.from_pretrained("jjonhwa/paragraph_s36000").to(device)
    retriever_tokenizer = AutoTokenizer.from_pretrained("jjonhwa/paragraph_s36000")

    if len(args.from_data.split("/")) <= 2: # `nickname/file_name` -->  HuggingFace Dataset
        data = load_dataset(args.from_data)
        train_data = data['train']
        train_data = pd.DataFrame(train_data)
    else: # Local File Path (RAW KO_WIKI_SUMMARY DATA를 활용할 경우 적용)
        train_data = pd.read_csv(args.from_data)
        train_data, count = preprocess.del_except(train_data)
        print(f"결과 없음 데이터의 개수: {count}")

        context, summary = preprocess.summary_parse(train_data)

        summary = [preprocessing(s) for s in summary]
        context = [preprocessing(c) for c in context]

        context, summary = preprocess.preprocess_length(
            context,
            summary,
            retriever_tokenizer,
            context_min_length=1500,
            summary_min_length=30
        )

        train_data = pd.DataFrame({"context": context, "summary": summary})

    train_data = train_data.iloc[:100]
    if 'summary' in train_data.columns:
        new_data = get_topn_passage_SUMMARY(
            retrieval,
            retriever_tokenizer,
            train_data,
            n=10,
            passage_length=200,
        )

        with open(os.path.join(args.save_folder, args.save_file), 'wb') as f:
            pickle.dump(new_data, f)

    else:
        new_data = get_topn_passage_QA(
            retrieval,
            retriever_tokenizer,
            train_data,
            n=10,
            passage_length=200,
        )

        with open(os.path.join(args.save_folder, args.save_file), 'wb') as f:
            pickle.dump(new_data, f)