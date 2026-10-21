# Conclusion (GRPO Pipeline Lab)


---

## 1. Lecture Overview

- The lesson builds a **pipeline for Group Relative Policy Optimization (GRPO)**, described as one of the popular **online RL** methods.
- Recap of online RL: the idea is to **let the model itself explore better responses**.
- The end-to-end workflow in the lab:
  1. Curate a set of **math problems**.
  2. Send them to a current language model and let it **generate multiple responses** per prompt.
  3. Curate a **reward function**: a *verifiable reward* that checks whether the response matches the ground truth.
  4. Obtain a **tuple of (prompt, responses, reward)**.
  5. Use **GRPO to update the language model**.

---

## 2. Setup

### Libraries
- Imports are very similar to the earlier DPO and SFT lessons.
- The difference: in **TRL**, this lab uses **`GRPOTrainer`** and **`GRPOConfig`** to set up the GRPO training environment.

### Evaluation dataset
- Unlike the previous two coding lessons (which tested the model on only a few example prompts), this lesson prepares a proper **evaluation dataset for math: GSM8K**.

### GPU flag
- A "use GPU" flag is set to **False** by default; set it to **True** if running on your own GPU machine.

### Persistent system prompt
- The prompt tells the model: *you are a helpful assistant that solves problems step by step, and always includes a final numeric answer inside a box (`\boxed{}`)*.
- This sentence is **critical**: it makes the model output the final response in a good format so the answer can be **easily extracted and compared** with the ground truth.

---

## 3. Reward Function

The reward function is used for **both** GRPO training (online RL) **and** evaluation on GSM8K.

### Inputs
- The model's **generated completions**
- The **ground truth**

### Logic
1. Use a **regular expression match** to capture the content inside the `boxed` (as instructed in the system prompt).
2. Take the **very first match** found and extract its content as the model's answer.
3. If there is **no match**, the model's extracted output is set to **empty**.
4. **Directly compare** the extracted content with the ground truth.
5. **Reward = 1** if identical; **Reward = 0** otherwise.

### Sanity tests of the reward function
| Case | Prediction | Ground truth | Reward |
|---|---|---|---|
| Positive sample | Reasoning steps followed by final answer boxed **72** | 72 | **1** |
| Negative sample (off by one) | Content inside box is **71** | 72 | **0** |

---

## 4. Preparing the Evaluation Dataset (GSM8K)

- Load the dataset from **OpenAI GSM8K**, using the **test** portion.
- Select a small number of samples (the lecture sets this to **5**) to speed up the process.
- Raw dataset structure: **questions** with **answers** that serve as ground truth.
- The final numeric answer in the raw data is **hidden after the "####" marker**, so it must be **extracted** as the ground truth.

### Post-processing function
- Matches the answer from after the marker, and sets the **ground truth** to the next item.
- Builds a **prompt** that combines:
  - the **system prompt** (instructing the model to put the answer in a box), and
  - the **user prompt** (the question itself).
- The function is mapped over the dataset to produce the new evaluation dataset.

### Resulting dataset format
Only **two columns**:
1. **`ground_truth`** — the numeric answer extracted from the original response.
2. **`prompt`** — the system prompt followed by the question.

---

## 5. Baseline Evaluation (Before Training)

### Model and procedure
- Model: **Qwen 2.5-0.5B Instruct**, evaluated on **5 prompts** from the GSM8K test set.
- Procedure:
  1. Start with empty lists of predictions and ground-truth labels.
  2. Loop through the post-processed dataset, taking the input prompt and ground truth.
  3. Generate a response with a generate-responses function (fed the model, tokenizer, and full message).
  4. Append the predictions and labels; print the response and ground truth for inspection.
  5. Use the reward function to count how many responses match the ground truth.
  6. Report **accuracy**.

### Results for the 5 prompts
| # | Outcome | Reason |
|---|---|---|
| 1 | No match | No boxed answer provided, so the answer couldn't be extracted — model not fully following the format |
| 2 | **Match** | Model boxed **3**, matching the ground truth |
| 3 | No match | Model didn't finish due to the **token limit** |
| 4 | No match | Model boxed **180**, which doesn't match the ground truth |
| 5 | No match | Model didn't finish; ground truth is **20** |

- **Baseline accuracy: 1 out of 5 = 20%.**

### Practical recommendations
- Allow a **much larger maximum number of tokens** in generation.
- Evaluate on the **full dataset**, because evaluating on only a few samples can come with **very large variance**.

---

## 6. Preparing the Training Dataset

- Evaluation is designed first; the fully trained model's evaluation is left to the end.
- Load GSM8K again, this time the **train** portion (split from the test portion).
- Apply the **same post-processing function** and **remove unnecessary columns**.
- If **not using a GPU**, select only the **first 10 ground truths** for training.
- Inspect the first example to see how the ground truth and prompt look.

---

## 7. GRPO Training Configuration

### `GRPOConfig` includes
- Batch-size-related hyperparameters
- Number of **epochs**
- **Learning rate**
- **Logging steps**
- **`num_generations`** — the key GRPO-specific hyperparameter

### Key hyperparameter: number of generations
- In GRPO, **multiple responses are generated for the same prompt**.
- `num_generations` controls **how many responses are generated per prompt**.
- In the lab it is set to **4** to speed up training.
- In practice it can be set **as high as 64 or even 128**, so there are **diverse enough responses** to **compare within the group**.

### Kicking off training
- Required components: **GRPO config**, **training dataset**, **reward function**.
- Pass the **model config, reward function, and train dataset** to **`GRPOTrainer`**.
- Because training a 0.5B model on a CPU machine can take very long, the lab uses a **HuggingFace small model** to speed up the process.
- Training takes a long time, so it is sped up in post-editing.

---

## 8. Training Observation: Zero Loss

- After training, the **training loss is always zero**.
- **Reason**: the starting model is **very small** and **cannot get most questions correct**.
- Consequently, in GRPO the **relative reward is all zero**, since the model never gets the answers correct.
- With a **larger model (Qwen 2.5B, as stated in the lecture)**, you would see a **meaningful training loss** and **meaningful improvement** during GRPO training.

---

## 9. Evaluation of the Fully Trained Model

### Setup
- A flag for the "fully trained Qwen" is set to **True** to load a model previously trained with **more resources (GPUs)**, with **slightly different config parameters**.
- Set it to **False** to evaluate the **HuggingFace small LLM** trained by the small GRPO trainer on the smaller dataset.

### Results for the 5 prompts
| # | Outcome | Details |
|---|---|---|
| 1 | No match | Boxed **20**, ground truth **18** |
| 2 | **Match** | Boxed **3**, ground truth 3 |
| 3 | No match | Still didn't finish |
| 4 | **Match** | Model correctly produced **540** |
| 5 | No match | Boxed **40**, ground truth **20** |

- **Trained model accuracy: 2 out of 5 = 40%** (vs. 20% baseline).

### Caveat
- For a fully meaningful comparison between the trained and previous model, **run the entire GSM8K test set**, not just the 5 samples.

---

## 10. Summary of the Lesson

The lesson walked through the whole process of:
1. **Building a math evaluation dataset**
2. **Creating / designing the reward function** for a math dataset
3. **Designing the evaluation process**
4. **Running the GRPO training cycle** on top of an existing **instruct model** (Qwen) to **improve its math capability**

The small-model run used a very small dataset to allow seeing the full GRPO training without long waiting times, given limited computational resources.

---

## 11. Quick Revision Checklist

- [ ] GRPO = a popular **online RL** method; the model explores better responses itself
- [ ] Pipeline: math prompts → multiple responses → verifiable reward → (prompt, responses, reward) → GRPO update
- [ ] TRL classes: `GRPOTrainer`, `GRPOConfig`
- [ ] System prompt asks for step-by-step solutions with the final numeric answer in `\boxed{}`
- [ ] Reward: regex-extract first boxed content; **1** if equals ground truth, else **0**; empty if no match
- [ ] GSM8K ground truth extracted from after the `####` marker
- [ ] Processed dataset columns: `ground_truth`, `prompt`
- [ ] `num_generations`: 4 in the lab; 64–128 in practice for diverse responses to compare within a group
- [ ] Zero training loss with the tiny model → relative rewards all zero (model never correct)
- [ ] Baseline accuracy 20% (1/5) → trained model 40% (2/5) on 5 samples only
- [ ] Use larger max tokens and the full test set for reliable evaluation
