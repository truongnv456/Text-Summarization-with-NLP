import yaml
import os
from general_utils import *

from transformers import RobertaTokenizer, EncoderDecoderModel, AutoTokenizer
import pickle


with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)


# v_loc: close
# Get the checkpoints from gcp
# os.makedirs(configs['output_dir']+'/pretrained/', exist_ok=True)
# os.system('gsutil -m cp -r "{}/*" "{}"'.format(configs['gcp_pretrained_path'],configs['output_dir']+'/pretrained/'))

test_data  = get_data_batch(path='./data/test_tokenized/*', test = True)
modelName = configs['output_dir']+'/checkpoint-8000/'
print("modelName", modelName)
model = EncoderDecoderModel.from_pretrained(modelName)
model.to("cuda")

batch_size = configs['batch_size'] * 2  # change to 64 for full evaluation

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = tokenizer(batch["original"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, 
                             attention_mask=attention_mask,
                             max_length = configs['decoder_max_length'],
                             early_stopping= configs['early_stopping'],
                             num_beams= configs['num_beams'], 
                             no_repeat_ngram_size= configs['no_repeat_ngram_size'])

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["original"])

rouge_output = rouge.compute(predictions= results["pred"], references= results["summary"] , rouge_types=["rouge1","rouge2","rougeL"])

os.makedirs('./testing/', exist_ok = True)
with open('./testing/prediction.pkl', 'wb') as f:
    pickle.dump(results["pred"], f, protocol=pickle.HIGHEST_PROTOCOL)
with open('./testing/reference.pkl', 'wb') as f:
    pickle.dump(results["summary"], f, protocol=pickle.HIGHEST_PROTOCOL)
with open('./testing/rouge.txt', 'w+') as f:
    for key,value in rouge_output.items():
        f.write(key.upper())
        f.write(' : ')
        f.write(repr(value.mid))
        f.write('\n')

pred_str = results["pred"]
label_str = results["summary"]
# v_loc: result:

cntTrue =  0
cntFalse =  0
for i in range(len(pred_str)):
    print("Nhan " + str(i))
    print(label_str[i])
    print("Du Doan " + str(i))
    print(pred_str[i])
    ret = label_str[i].strip() == pred_str[i].strip()
    if ret:
        print("--------------------> ", ret)
        cntTrue += 1
    else:
        cntFalse += 1

print("true--> ", cntTrue)
print("false--> ", cntFalse)
 