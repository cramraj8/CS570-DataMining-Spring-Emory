
import json
import os
import time
import datetime
from collections import defaultdict

import torch
from bunch import Bunch
from pytorch_transformers import BertTokenizer, BertModel, WarmupLinearSchedule, AdamW
from dataset import TrainTRECDataset, TestTRECDataset
from model import TRECCARModel
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import pandas as pd
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

CONFIG_FILE = "config.json"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def get_config_from_json(json_file):
    """
        Get the config from a json file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def format_time(elapsed_time):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed_time)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


config, _ = get_config_from_json(CONFIG_FILE)
seed_val = config.cmd_args['seed']
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
os.makedirs(config.data['results_dir'], exist_ok=True)

# Loading Tokenizer
tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"], cache_dir=config.data['pretrained_download_dir'])
dataset = TrainTRECDataset(config.data['train_data'], config, is_train=True, bert_tokenizer=tokenizer)
train_dataloader = DataLoader(dataset=dataset,
                              batch_size=config.training["train_batch_size"],
                              pin_memory=config.cmd_args['device'] == 'cuda:0',
                              num_workers=config.training['num_workers'],
                              shuffle=True)
n_train_batches = len(train_dataloader)
print("Number of train batches : ", n_train_batches)

# Creating instance of BertModel
net = TRECCARModel(config, freeze_bert=True)
net.to(device)

criterion = nn.MarginRankingLoss(margin=1, size_average=True)
opti = AdamW(net.parameters(),
             lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
             eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
             correct_bias=False
             )
# opti = optim.Adam(net.parameters(), lr=2e-5)

# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

num_epochs = config.training['epochs']
display_step = config['training']['display_step']
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(opti,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)
# scheduler = WarmupLinearSchedule(opti, warmup_steps=config.training["warmup_proportion"],
#                                  t_total=config.training["total_training_steps"])
total_t0 = time.time()
training_stats = []
history = defaultdict(list)
for epoch_idx in range(0, num_epochs):
    # ========================================
    #               Training
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_idx + 1, num_epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    net.train()  # TODO: IMPORTANT !
    for batch_idx, train_batch_data in enumerate(train_dataloader):
        # Clear gradients
        net.zero_grad()  # TODO: check validity !
        opti.zero_grad()

        # Converting these to cuda tensors
        pos_ids, pos_mask, pos_type_ids, \
        neg_ids, neg_mask, neg_type_ids, \
        seqA_len, posSeqB_len, negSeqB_len, \
        label = train_batch_data

        pos_ids, pos_mask, pos_type_ids, \
        neg_ids, neg_mask, neg_type_ids, \
        seqA_len, posSeqB_len, negSeqB_len, \
        label = pos_ids.to(device), pos_mask.to(device), pos_type_ids.to(device), \
                neg_ids.to(device), neg_mask.to(device), neg_type_ids.to(device), \
                seqA_len.to(device), posSeqB_len.to(device), negSeqB_len.to(device), \
                label.to(device)

        pos_net_output = net(pos_ids, attn_masks=pos_mask, type_ids=pos_type_ids)
        neg_net_output = net(neg_ids, attn_masks=neg_mask, type_ids=neg_type_ids)
        # # TODO: do i need a softmax or not ?

        # Computing loss
        # loss = criterion(net_output, label.float())
        loss = criterion(pos_net_output, neg_net_output, label.float())
        # total_train_loss += loss.item()

        # Back propagating the gradients
        loss.backward()
        if config.training['gradient_clipping']['use']:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.training['gradient_clipping']['clip_value'])

        # Optimization step
        opti.step()

        # Progress update every display_step batches.
        if batch_idx % display_step == 0 and not batch_idx == 0:
            elapsed = format_time(time.time() - t0)
            # print('  Batch {:>5,}  of  {:>5,}  :  loss - {:>5,.2f}    Elapsed: {:}.'.format(batch_idx,
            #                                                                                 len(train_dataloader),
            #                                                                                 loss, elapsed))
            print('  Epoch {:>5,}  of  {:>5,}  :  Batch {:>5,}  of  {:>5,}  :  \
            loss - {:>5,.2f}    Elapsed: {:}.'.format(epoch_idx + 1, num_epochs,
                                                      batch_idx + 1, len(train_dataloader),
                                                      loss, elapsed))
            training_stats.append(
                {
                    'epoch': epoch_idx + 1,
                    'batch': batch_idx + 1,
                    'step': (epoch_idx * n_train_batches) + batch_idx + 1,
                    'Training Loss': loss,
                    # 'Training Loss': avg_train_loss,
                    # 'Training Time': training_time,
                }
            )

    scheduler.step()  # TODO: IMPORTANT !

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

history_df = pd.DataFrame({"step": [e['step'] for e in training_stats],
                           "Training Loss": [e['Training Loss'] for e in training_stats]})
# history_df.to_csv(os.path.join(config.data['results_dir'], "history.csv"), index=False)

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

if not config.cmd_args['mode'] == "experiment":
    torch.save(net, os.path.join(config.data['results_dir'], "model-dump.pkl"))
# ========================================
#               NOT Validation, Just Testing
# ========================================
print("")
print("Validation...")
t0 = time.time()

test_dataset = TestTRECDataset(config.data['test_data'], config, is_train=False, bert_tokenizer=tokenizer)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=config.training["test_batch_size"],
                             pin_memory=config.cmd_args['device'] == 'cuda',
                             num_workers=config.training['num_workers'],
                             shuffle=True)
n_test_batches = len(test_dataloader)
print("Number of test batches : ", n_test_batches, "\n")
net.eval()

qID_list = []
paraID_list = []
pScore_list = []
t1 = time.time()
for batch_idx, test_batch_data in enumerate(test_dataloader):
    # Converting these to cuda tensors
    input_seq, input_mask, input_type_ids, label, qID, passageID, seqA_len, seqB_len = test_batch_data
    input_seq, input_mask, input_type_ids, \
    seqA_len, seqB_len = input_seq.to(device), input_mask.to(device), input_type_ids.to(device), \
                         seqA_len.to(device), seqB_len.to(device)

    with torch.no_grad():
        net_output = net(input_seq, attn_masks=input_mask, type_ids=input_type_ids)
        net_output = net_output.detach().cpu().numpy()

        for i in range(len(qID)):
            qID_list.append(qID[i])
            paraID_list.append(passageID[i])
            pScore_list.append(net_output[i])
    elapsed = format_time(time.time() - t1)
    print('  Batch {:>5,}  of  {:>5,}  :  processed    Elapsed: {:}.'.format(batch_idx,
                                                                             n_test_batches,
                                                                             elapsed))

pScore_list = [float(e) for e in pScore_list]
predicted_df = pd.DataFrame({"qID": qID_list,
                             "pID": paraID_list,
                             "pScore": pScore_list}, columns=["qID", "pID", "pScore"])
if not config.cmd_args['mode'] == "experiment":
    predicted_df.to_csv(os.path.join(config.data['results_dir'], "predictions.csv"))
print()

# ================================================
#               Reverse Sorting Relevance
# ================================================
predicted_df = predicted_df[['qID', 'pID', 'pScore']]
grouped_pred_df = predicted_df.groupby(["qID"])
num_queries = len(grouped_pred_df)
missing_q_sets = 0
save_ranked_file = os.path.join(config.data['results_dir'], "ranked.test.relevance.txt")
with open(save_ranked_file, 'w') as write_file:
    q_cnt = 1
    for name, row_group in grouped_pred_df:
        rank_cnt = 1

        # ======= SORTING =======
        sorted_row_group = row_group.sort_values(by='pScore', ascending=False, inplace=False)
        # =======================

        if len(sorted_row_group) != 100:
            # print(">>>>>>>>>>> Missing query %s with shape %s" % (name, sorted_row_group.shape))
            # print(">>>>>>>>>>> Missing query with size %s" % sorted_row_group.shape[0])
            missing_q_sets += 1

        for i, row in sorted_row_group.iterrows():
            write_file.write("%s\tQ0\t%s\t%s\t%s\trchan31\n" % \
                             (row["qID"], row["pID"], rank_cnt, row["pScore"]))
            rank_cnt += 1

        print("Finished composing for query number : %s / %s" % (q_cnt, num_queries))
        q_cnt += 1
print()
print("Missing query-doc pairs : ", missing_q_sets)
print("Done train, val, and test !!!")

"""
References:
1. https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
2. https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

"""

# # ================================================
# #               Run TREC-eval bash
# # ================================================
# QRELFile_path = "../../../Data/qrelsY1-test.V2.0/automatic-test.pages.cbor-hierarchical.qrels"
# save_eval_filepath = os.path.join(config.data['results_dir'], "eval.txt")
# cmd = "../../../trec_eval-master/trec_eval {} {} -m all_trec > {}".format(QRELFile_path, save_ranked_file,
#                                                                           save_eval_filepath)
# os.system(cmd)

