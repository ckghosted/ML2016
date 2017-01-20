codes are in src/
if you just want to fine tune, apply my_fine_tune.py to the previous output answers

Getting NN model  : python weights.py data_path feature_name
   weights.py is in src/
   if feat/feats.npy exits, it will automatically use the preprocessed feature to training.
   else, it will do from scratch and save the processed feature as feat/feats.npy
   if the feature_name file doesn't exits in feat/, it will run from beginning and save feature as feat/filename, however, this is used only as feat/feats.npy doesn't exit.
   for saving time, one can just run python weights.py ../data weights.npy
   The best model will be saved as /model/best.h5

from scratch:
   for the followings, if you adjust nb_cluster to be less than the file numbers in each directory, please remove all the files in each directory before running codes
   run cluster.py to get required files ready (can adjust nb_cluster in it, store in /clusters as csv format and the s_probs in /probs)
   run count_n_grams_fine_tune.py to get tags (store in /cluster_tags) (this is from the mainstream method)
   run merge.py to merge all answers in /cluster_tags to /merge_resluts/result.csv (this is the answer before fine-tuning)
   run my_fine_tune.py to fine-tune /merge_resluts/result.csv to /merge_resluts/tuned.csv