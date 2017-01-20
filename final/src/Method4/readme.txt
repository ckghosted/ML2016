[Method 4] Readme:
$ python freq_grams.py test.csv
(根據第一個參數去找 test.csv (物理文本), 在目前的目錄產生 test_est.csv, 為可以過weak baseline的答案)
(line 20: run_pos_tag = False若改為True, 則會產生pos_dict.csv, 為每個字的詞性統計)
(若第一個參數也可以是其他六類主題的文本, 例如biology.csv, 則會在目前的目錄產生 biology_est.csv 和 biology_pos_dict.csv)

$ python fine_tune.py 
(根據第一個參數去找 test_est.csv, 需要在同一個目錄放置該類別的詞性統計(例如pos_dict.csv), 程式會在目前的目錄產生test_est2.csv，為可以過strong baseline的答案)