# A. FIXME
1. data_len not correct in AMINO/datamodule/preprocess.py FFT. FIX!!!
2. logging do not print anything, should be hydra job problem.
3. exp_dir should be fix, should be hydra job problem.

# B. Test
1. test_step in AMINO/modules/autoencoder.py haven't been test.
2. logger.py in AMINO/callbacker haven't been test.
3. Batch_size > 1 haven't been test.

# C. Enhancement
1. some `eval` function should be replace by dynamic_import.
2. on-the-fly data-augment like specaug and speed should be added into preprocess

# D. Note
This file might overrides through https://marketplace.visualstudio.com/items?itemName=wayou.vscode-todo-highlight

# Priority: