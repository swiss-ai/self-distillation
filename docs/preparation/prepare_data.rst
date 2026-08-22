Prepare Data for Post-Training
========================================

Last updated: 02/09/2025.

Before starting the post-training job, we need to prepare the data for
the policy training. The data should be stored in the parquet format.

We provide several data preprocess scripts for different datasets,
including GSM8K, MATH, HelloSwag, Full_hh_rlhf. To prepare other datasets, we need
to follow the following steps: The data preprocess script can be divided
into two parts:

1. The first part is the common part, which loads the dataset from
   huggingface's ``datasets`` package. Then preprocess the datasets with
   the ``make_map_fn`` and then store in the parquet format.

.. code:: python

   import re
   import os
   import datasets

   from verl.utils.hdfs_io import copy, makedirs
   import argparse

   # To extract the solution for each prompts in the dataset
   # def extract_solution(solution_str): 
   # ...


   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--local_dir', default='/opt/tiger/gsm8k')
       parser.add_argument('--hdfs_dir', default=None)

       args = parser.parse_args()

       num_few_shot = 5
       data_source = 'openai/gsm8k'

       dataset = datasets.load_dataset(data_source, 'main')

       train_dataset = dataset['train']
       test_dataset = dataset['test']

           # Construct a `def make_map_fn(split)` for the corresponding datasets.
       # ...
           
       train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
       test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

       local_dir = args.local_dir
       hdfs_dir = args.hdfs_dir

       train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
       test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

       makedirs(hdfs_dir)

       copy(src=local_dir, dst=hdfs_dir)

2. The users are required to implement the ``make_map_fn()`` function
   (as well as the ``extract_solution``) on their own to support
   different datasets or tasks.

We already implemented the data preprocess of GSM8k, MATH, Hellaswag and Full_hh_rlhf
datasets. And we take the GSM8k dataset as an example:

**GSM8K**

In the ``make_map_fn``, each data field should consist of the following
4 fields:

1. ``prompt``: This field should be constructed in the format of
   huggingface chat_template. The tokenizer in ``RLHFDataset`` will
   apply chat template and tokenize the prompt.
2. ``extra_info``: Record metadata for the current prompt. The canonical
   field is ``extra_info["data_source"]``.
3. ``reward_model``: Stores the ``ground_truth`` and the canonical
   ``scoring_module`` to use for reward computation.
4. ``answer``: Optional privileged answer text used by teacher-based
   methods such as SDPO.

.. code:: python

   def extract_solution(solution_str):
       solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
       assert solution is not None
       final_solution = solution.group(0)
       final_solution = final_solution.split('#### ')[1].replace(',', '')
       return final_solution

   instruction_following = "Let's think step by step and output the final answer after \"####\"."

   # add a row to each data item that represents a unique id
   def make_map_fn(split):

       def process_fn(example, idx):
           question = example.pop('question')

           question = question + ' ' + instruction_following

           answer = example.pop('answer')
           solution = extract_solution(answer)
           data = {
               "prompt": [{
                   "role": "user",
                   "content": question
               }],
               "extra_info": {
                   "data_source": data_source,
                   'split': split,
                   'index': idx
               },
               "reward_model": {
                   "ground_truth": solution,
                   "scoring_module": "math"
               },
               "answer": answer
           }
           return data

       return process_fn
