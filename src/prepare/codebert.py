from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer
import torch

model_name = '/data/data_public/yrb/devign_lab/codebert-base'
tokenizer_name = '/data/data_public/yrb/devign_lab/codebert-base-tokenizer'

# 加载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = RobertaModel.from_pretrained(model_name).to(torch.device('cuda'))

def get_code_embedding(code):

    # 将代码进行 Tokenize
    inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True, max_length=512).to(torch.device('cuda'))

    # 获取模型输出（返回的包含多个部分，通常我们使用最后一层的 hidden states）
    with torch.no_grad():
        outputs = model(**inputs)

    # 通过模型输出获取代码的向量表示
    # 输出的 `last_hidden_state` 是一个 tensor，形状为 (batch_size, sequence_length, hidden_size)
    code_embedding = outputs.last_hidden_state.mean(dim=1)  # 取平均得到句子的向量表示

    return code_embedding
