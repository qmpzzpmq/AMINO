# A. FIXME
1. data_len not correct in AMINO/datamodule/preprocess.py FFT. # FIX !!!
2. logging do not print anything, should be hydra job problem. # FIX !!!
3. exp_dir should be fix, should be hydra job problem. # FIX !!!
4. Hydra log file is not correct.
5. Wandb should log hyper-parameter.

# B. Test
1. test_step in AMINO/modules/autoencoder.py haven't been test. # FIX !!!
2. logger.py in AMINO/callbacker haven't been test. # FIX !!!
3. Batch_size > 1 haven't been test. # Fix !!! but it seems all audio same length

# C. Enhancement
1. some `eval` function should be replace by dynamic_import. # FIX!!! except scheduler
2. on-the-fly data-augment like specaug and speed should be added into preprocess
3. more feature should be support
4. more net should be support

# D. Performance
1. the anormal loss lower than normal # FIX ！！！

# E. Note
This file might overrides through https://marketplace.visualstudio.com/items?itemName=wayou.vscode-todo-highlight

# Priority: