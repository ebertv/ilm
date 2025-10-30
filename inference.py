GPT_MODEL_DIR = '.ilm/train'
LLAMA_MODEL_DIR = 'ilm/llama1b'
MASK_CLS = 'ilm.mask.custom.MaskTurns'

from argparse import ArgumentParser
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
import os
import pickle
import ilm.tokenize_util
from ilm.infer import infill_with_ilm
from train_ilm import TargetType
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import json
from tqdm import tqdm
from train_ilm import Task

parser = ArgumentParser()

parser.add_argument('tag', type=str)
parser.add_argument('--examples_dir', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--max_num_preview', type=int)

task_args = parser.add_argument_group('Task')
task_args.add_argument('--task', type=str, choices=[t.name.lower() for t in Task])

data_args = parser.add_argument_group('Data')
data_args.add_argument('--data_no_cache', action='store_false', dest='data_cache')
data_args.add_argument('--data_loader_num_workers', type=int)

eval_args = parser.add_argument_group('Eval')
eval_args.add_argument('--eval_only', action='store_true', dest='eval_only')
eval_args.add_argument('--eval_examples_tag', type=str)
eval_args.add_argument('--eval_max_num_examples', type=int)
eval_args.add_argument('--eval_batch_size', type=int)
eval_args.add_argument('--eval_sequence_length', type=int)
eval_args.add_argument('--eval_skip_naive_incomplete', action='store_true', dest='eval_skip_naive_incomplete')

parser.add_argument('--model_type', type=str, choices=['gpt2', 'llama'], default='gpt2')

parser.set_defaults(
    examples_dir=None,
    seed=None,
    max_num_preview=8,
    task='ilm',
    data_cache=True,
    data_loader_num_workers=4,
    eval_only=False,
    eval_examples_tag='test',
    eval_max_num_examples=None,
    eval_batch_size=8,
    eval_sequence_length=256,
    eval_skip_naive_incomplete=False)



args = parser.parse_args()

if args.model_type == 'gpt2':
    MODEL_DIR = GPT_MODEL_DIR
    assert args.eval_sequence_length <= 1024
    model_type = GPT2LMHeadModel
    cfg_type = GPT2Config
    tokenizer = ilm.tokenize_util.Tokenizer.GPT2
elif args.model_type == 'llama':
    MODEL_DIR = LLAMA_MODEL_DIR
    assert args.eval_sequence_length <= 4096
    from transformers import LlamaForCausalLM, LlamaConfig
    model_type = LlamaForCausalLM
    cfg_type = LlamaConfig
    tokenizer = ilm.tokenize_util.Tokenizer.LLAMA

model = model_type.from_pretrained(MODEL_DIR)
model.eval()
model.cuda()

with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
try:
    ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')
print(additional_tokens_to_ids)

out_fn_to_fp = lambda fn: os.path.join(MODEL_DIR, fn)

print('Loading eval data')
if args.examples_dir is None:
    fp = args.tag
else:
    fp = os.path.join(args.examples_dir, '{}.pkl'.format(args.tag))
with open(fp, 'rb') as f:
    masked_data = pickle.load(f)
import numpy as np
from train_ilm import masked_dataset_to_inputs_and_tts
import ilm.mask.util
base_vocab_size = ilm.tokenize_util.vocab_size(tokenizer)
start_infill_id = additional_tokens_to_ids["<|startofinfill|>"]
end_infill_id = additional_tokens_to_ids["<|endofinfill|>"]
mask_cls = ilm.mask.util.mask_cls_str_to_type(MASK_CLS)
mask_types = mask_cls.mask_types()
mask_type_to_id = {}
for i, t in enumerate(mask_types):
    if args.model_type == 'gpt2':
        t_id = 50259 #for GPT2
    elif args.model_type == 'llama':
        t_id = 128258 #for LLaMA
    t_tok = '<|infill_{}|>'.format(mask_cls.mask_type_serialize(t))
    additional_ids_to_tokens[t_id] = t_tok
    mask_type_to_id[t] = t_id
eval_inputs, eval_tts, eval_num_docs = masked_dataset_to_inputs_and_tts(
                                        'test',
                                        tokenizer,
                                        start_infill_id,
                                        end_infill_id,
                                        mask_type_to_id,
                                        args)

print('Loaded {} eval examples from {} documents'.format(len(eval_inputs), eval_num_docs))

eval_tt_to_count = {TargetType(k):v for k, v in zip(*np.unique(eval_tts, return_counts=True))}
num_unmasked = eval_tt_to_count.get(TargetType.CONTEXT, 0)
num_masked = eval_tt_to_count.get(TargetType.INFILL, 0)
print('Mask rate (tokens): {:.4f}'.format(num_masked / (num_unmasked + num_masked)))
print('{} documents, {} examples'.format(eval_num_docs, eval_inputs.shape[0]))
eval_data = TensorDataset(
    torch.from_numpy(eval_inputs.astype(np.int64)),
    torch.from_numpy(eval_tts))
del eval_inputs
del eval_tts

eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8, drop_last=True)

print('Starting inference with model from {}'.format(MODEL_DIR))
errors = 0
for i, eval_batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
    with torch.no_grad():
        eval_inputs, eval_tts = tuple(t for t in eval_batch)
        for j in range(len(eval_inputs)):
            #remove everything after the first sep token
            context = eval_inputs[j].tolist()
            start_infill_token = additional_tokens_to_ids["<|startofinfill|>"]
            end_infill_token = additional_tokens_to_ids["<|endofinfill|>"]
            turn_infill_token = additional_tokens_to_ids["<|infill_turn|>"]
            if start_infill_token in context:
                if end_infill_token not in context:
                    print('Malformed example, has start_infill_token but not end_infill_token')
                    print(f'Context: {ilm.tokenize_util.decode(context, tokenizer)}')
                    print('Skipping')
                    errors += 1
                    continue
                real = context[context.index(start_infill_token)+1:context.index(end_infill_token)]
                context = context[:context.index(start_infill_token)]
                # replace turn_infill_token with real in context
                real_context = []
                for t in context:
                    if t == turn_infill_token:
                        real_context.extend(real)
                    else:
                        real_context.append(t)
            try:
                generated = infill_with_ilm(
                    model,
                    additional_tokens_to_ids,
                    context,
                    num_infills=1)
            except Exception as error:
                print('Error during inference:')
                errors += 1
                print(error)

            for g in generated:
                sample = {
                    'batch_idx': i,
                    'example_idx': j,
                    'full_context': ilm.tokenize_util.decode(real_context, tokenizer),
                    'real_infill': ilm.tokenize_util.decode(real, tokenizer),
                    'generated_infill': ilm.tokenize_util.decode(g[0], tokenizer)
                }
                output_fp = f'ilm_{args.model_type}_infill_{args.tag}.jsonl'
                output_fp = os.path.join(args.examples_dir, output_fp)
                with open(output_fp, 'ab') as f:
                    f.write((json.dumps(sample) + '\n').encode('utf-8'))
                
if errors > 0:
    print(f'Finished with {errors} errors')
else:
    print('Finished successfully')
    