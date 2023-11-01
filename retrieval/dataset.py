import torch

from torch.utils.data import Dataset

class Cross_Dataset(Dataset):
    def __init__(self, args, query, document, tokenizer):
        self.query = query
        self.document = document
        self.tokenizer = tokenizer
        self.args = args

        if self.args.version == 1:
            self.max_length = 300
            self.stride = 128
            self.type = 'paragraph'
        elif self.args.version == 2:
            self.max_length = 2048
            self.stride = 512
            self.type = 'paragraph'
        elif self.args.version == 3:
            self.max_length = 512
            self.stride = 128
            self.type = 'chunk'
        elif self.args.version == 4:
            self.max_length = 2048
            self.stride = 512
            self.type = 'chunk'

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        question = self.query[idx]
        doc = self.document[idx]

        if self.args.model == 'klue/bert-base':
            encoded_example = self.tokenizer(
                question,
                doc,
                truncation='only_second',
                max_length=self.max_length,
                stride=self.stride,
                padding='max_length',
                # return_token_type_ids=False,
                return_tensors='pt'
            )
            
            train_dataset = torch.cat([
                encoded_example['input_ids'],
                encoded_example['attention_mask'],
                encoded_example['token_type_ids'],
            ])

        else:
            encoded_example = self.tokenizer(
                question,
                doc,
                truncation='only_second',
                max_length=self.max_length,
                stride=self.stride,
                padding='max_length',
                return_tensors='pt'
            )

            train_dataset = torch.cat([
                encoded_example['input_ids'],
                encoded_example['attention_mask'],
            ])
            
        return train_dataset
    
class Cross_Dataset_Para(Dataset):
    def __init__(self, args, query, document, label, tokenizer):
        self.query = query
        self.document = document
        self.tokenizer = tokenizer
        self.label = torch.tensor(label)
        self.args = args

        if self.args.version == 1:
            self.max_length = 300
            self.stride = 128
            self.type = 'paragraph'
        elif self.args.version == 2:
            self.max_length = 2048
            self.stride = 512
            self.type = 'paragraph'
        elif self.args.version == 3:
            self.max_length = 512
            self.stride = 128
            self.type = 'chunk'
        elif self.args.version == 4:
            self.max_length = 2048
            self.stride = 512
            self.type = 'chunk'

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        question = self.query[idx]
        doc = self.document[idx]
        label = self.label[idx]
        
        encoded_example = self.tokenizer(
            question,
            doc,
            truncation='only_second',
            max_length=self.max_length,
            stride=self.stride,
            padding='max_length',
            # return_token_type_ids=False,
            return_tensors='pt'
        )
        
        encoded_example['input_ids'] = encoded_example['input_ids'].squeeze(0)
        encoded_example['attention_mask'] = encoded_example['attention_mask'].squeeze(0)
        encoded_example['token_type_ids'] = encoded_example['token_type_ids'].squeeze(0)
        encoded_example['label'] = label

        return encoded_example