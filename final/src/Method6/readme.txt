[Method 6] Readme:
$ python lsa_kmeans.py test.csv test_est2.csv
(根據前兩個參數去找 test.csv (物理文本) 以及 test_est2.csv (一份kaggle上面表現不錯的答案)，產生km_labels_0, km_labels_1, km_labels_2, 和km_labels_3, 為分群的結果)

$ python add_cluster_tags.py test.csv test_est2.csv
(根據前兩個參數去找 test.csv (物理文本) 以及 test_est2.csv (一份kaggle上面表現不錯的答案), 以及同目錄之下的km_labels_0, km_labels_1, km_labels_2, km_labels_3 (分群的結果), 產生test_est3.csv, 為Kaggle best)