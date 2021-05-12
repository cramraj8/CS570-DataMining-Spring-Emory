import json
import glob
import os


ROOT_FOLDER = "/raid6/home/ramraj/2021/ir/Contextual-Reranking/Data/"
BENCHMARK_TRAIN_FOLD_FOLDER = os.path.join(ROOT_FOLDER, "benchmark-train-relevance-v2.0")
SRC_RAW_ENTITIES_FOLDER = "/raid6/home/ramraj/2021/ir/entity-reranking/Entity-Linking/Intermediate-Data/Raw-Entities-Train/"
DST_DIR = "/raid6/home/ramraj/2021/ir/entity-reranking/Entity-Linking/Train-with-entities/"

if not os.path.exists(os.path.join(DST_DIR)):
    os.makedirs(os.path.join(DST_DIR))


if __name__ == "__main__":
    src_files = sorted(glob.glob(os.path.join(BENCHMARK_TRAIN_FOLD_FOLDER, "fold-*-train.pages.cbor-hierarchical.benchmark.json")))
    print(len(src_files))
    for file_idx, src_file in enumerate(src_files):
        print("Start processing fold -", file_idx)
        fold_idx = src_file.split("/")[-1].split("-")[1]
        save_fold_filename = os.path.join(DST_DIR, "fold-{}.json".format(fold_idx))
        
        data = json.load(open(src_file, 'r'))
        
        for q_idx, data_sample in enumerate(data):
            q_text = data_sample['qString']
            saved_q_filename = os.path.join(SRC_RAW_ENTITIES_FOLDER, "fold-{}".format(fold_idx), "query", "{}.json".format(q_idx))
            
            q_entites = json.load(open(saved_q_filename, 'r'))
            data[q_idx]['qEntities'] = q_entites
            

            for d_idx, rel_docs in enumerate(data_sample['RelevantDocuments']):
                doc_text = rel_docs['docText']
                saved_d_filename = os.path.join(SRC_RAW_ENTITIES_FOLDER, "fold-{}".format(fold_idx), "doc", "{}_{}.json".format(q_idx, d_idx))
                                
                d_entites = json.load(open(saved_d_filename, 'r'))
                data[q_idx]['RelevantDocuments'][d_idx]['dEntities'] = d_entites


                
#             if q_idx  == 10:
#                 break
                
        # SAVE JSON
        with open(save_fold_filename, 'w') as save_file:
            json.dump(data, save_file, indent=4)

        # break  # for folds


 