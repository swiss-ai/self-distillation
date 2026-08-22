# Dataset Format
## RLHF dataset
We combine all the data sources into a single parquet files. We directly organize the prompt into the chat format so that multi-turn chats can be easily incorporated. In the prompt, we may add instruction following texts to guide the model output the answers in a particular format so that we can extract the answers.

Math problems
```json
{
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "extra_info": {
        "data_source": "openai/gsm8k"
    },
    "reward_model": {
        "ground_truth": "72",
        "scoring_module": "math"
    },
    "answer": "Natalia sold 48 + 24 = 72 clips altogether. #### 72"
}
```
