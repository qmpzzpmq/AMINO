# FIXME
1. data_len not correct in AMINO/datamodule/preprocess.py FFT.
2. logging do not print anything, should be hydra job problem.
3. exp_dir should be fix, should be hydra job problem.

# Test
1. test_step in AMINO/modules/autoencoder.py haven't been test.
2. logger.py in AMINO/callbacker haven't been test.
3. Batch_size > 1 haven't been test.

# Enhancement
1. some `eval` function should be replace by dynamic_import.

# Note
This file might overrides through https://marketplace.visualstudio.com/items?itemName=wayou.vscode-todo-highlight