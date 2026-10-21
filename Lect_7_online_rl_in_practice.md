# Post-training of LLMs — Lecture: Online RL in Practice


---

## 1. Lecture Overview

- The lesson builds a pipeline for **Group Relative Policy Optimization (GRPO)**, described as one of the popular online RL methods.
- Recap: **online reinforcement learning** lets the model itself explore better responses.
- Lab workflow:
  1. Curate a set of **math problems**.
  2. Send them to a current language model and let it **generate multiple responses** per prompt.
  3. Curate a **reward function** — a *verifiable reward* that checks whether the response matches the ground truth.
  4. Obtain tuples of **(prompt, response, reward)**.
  5. Use **GRPO to update the language model**.

---

## 2. Setup

- Imports are very similar to the earlier DPO and SFT lessons.
- The difference in TRL: use **`GRPOTrainer`** and **`GRPOConfig`** to set up the GRPO training environment.
- Unlike the previous two coding lessons (tested on a few example prompts), this lab prepares a proper **evaluation dataset for math: GSM8K**.
- A `use GPU` flag is set to **false** by default; set it to true when running on your own GPU machine.

### System prompt
- A persistent system prompt: *"You are a helpful assistant that solves problems step by step, and always includes a final numeric answer inside a boxed."*
- This sentence is **critical**: it makes the model output the final response in a consistent format so the answer can be **easily extracted and compared** with the ground truth.

---

## 3. Reward Function (Verifiable Reward)

Used for **both** online RL training and GSM8K evaluation.

**Inputs:** the model's generated completions and the ground truth.

**Logic:**
1. Use a **regular expression** to capture the content inside `boxed`, as instructed by the system prompt.
2. Among all matches, take the **very first match** as the model's answer.
3. If there is **no match**, set the model's output to **empty**.
4. Directly compare the extracted content with the ground truth.
5. **Reward = 1** if identical; **Reward = 0** otherwise.

**Sanity tests:**
| Case | Boxed answer | Ground truth | Reward |
|---|---|---|---|
| Positive sample | 72 | 72 | 1 |
| Negative sample (off by one) | 71 | 72 | 0 |

---

## 4. Evaluation Dataset Preparation (GSM8K)

- Load from **OpenAI GSM8K**, using the **test** portion.
- Select a small subset (**5 samples**) to speed up the process.
- Raw dataset: questions plus answers; the final answer is hidden **after the `####` marker** in the answer text.
- **Post-processing function:**
  - Match and extract the answer after the marker and set it as the **ground truth**.
  - Rebuild the **prompt** = system prompt (instructs boxed answer) + user prompt (the question).
- The processed dataset has **two columns**:
  - `ground truth` — the number extracted from the original answer.
  - `prompt` — system prompt followed by the question.

---

## 5. Baseline Evaluation

- Model: **Qwen 2.5-0.5B Instruct**, evaluated on 5 GSM8K test prompts.
- Procedure:
  - Start with empty lists of predictions and ground-truth labels.
  - Loop through the processed dataset; for each prompt, generate a response with the generate-response function (model, tokenizer, full message).
  - Append predictions and labels, print them for inspection.
  - Use the reward function to count matches and report **accuracy**.

### Baseline results (5 samples)
| # | Observation | Match? |
|---|---|---|
| 1 | No boxed answer in output → nothing extracted | No |
| 2 | Boxed 3 | Yes |
| 3 | Model didn't finish (token limit) | No |
| 4 | Boxed 180, wrong | No |
| 5 | Model didn't finish; ground truth is 20 | No |

- **Accuracy: 1/5 = 20%**

### Practical recommendations
- Allow a **much larger maximum number of generation tokens**.
- Evaluate on the **full dataset**, since a few samples come with **very large variance**.

---

## 6. Training Dataset Preparation

- Load GSM8K again, using the **train** split.
- Apply the **same post-processing function** and remove unnecessary columns.
- When **not using GPU**, select only the **first 10** ground truths for training.
- Inspect the first example to see how ground truth and prompt look.

---

## 7. GRPO Configuration

`GRPOConfig` includes:
- Batch-size-related hyperparameters
- Number of epochs
- Learning rate
- Logging steps

### Key GRPO-specific hyperparameter: `num_generations`
- In GRPO, **multiple responses are generated for the same prompt**; this parameter controls how many.
- Set to **4** in the lab to speed up training.
- In practice it can be set as high as **64 or even 128** to get **diverse enough responses**, so responses can be **compared within the group**.

---

## 8. Running GRPO Training

- Ingredients ready: GRPO config, training dataset, reward function.
- Training Qwen 0.5B on a CPU machine takes very long, so a **small HuggingFace model** is used in the lab to speed things up.
- Pass the **model config, reward function, and train dataset** to `GRPOTrainer` and start training.

### Why training loss was always zero
- The lab starts from a **very small model that cannot get most questions correct**.
- In GRPO the **relative reward is all zero** since the model never gets answers correct.
- With a **larger model (e.g., Qwen 2.5B)**, you'll see **meaningful training loss** and **meaningful improvement**.

---

## 9. Evaluating the Fully Trained Model

- A `fully trained Qwen` flag set to **true** loads a model previously trained with **more resources (GPUs)** and **slightly different config parameters**.
- Set it to false to evaluate the small HuggingFace model trained on the small dataset.

### Results (5 samples)
| # | Observation | Match? |
|---|---|---|
| 1 | Boxed 20, ground truth 18 | No |
| 2 | Boxed 3, ground truth 3 | Yes |
| 3 | Didn't finish | No |
| 4 | Boxed 540 | Yes (correct) |
| 5 | Boxed 40, ground truth 20 | No |

- **Accuracy: 2/5 = 40%** (up from 20% baseline)
- For a **fully meaningful comparison** between trained and previous models, run the **entire GSM8K test set** rather than only five samples.

---

## 10. Key Takeaways

1. **Verifiable rewards** (answer-matching against ground truth) make online RL straightforward for math.
2. A **consistent output format** (boxed final answer via system prompt) is essential for reliable answer extraction and reward computation.
3. The same reward function serves both **training** and **evaluation**.
4. GRPO samples **multiple responses per prompt** (`num_generations`) and compares them **relative to each other within the group**.
5. Larger `num_generations` (up to 64–128) gives more diverse responses to compare.
6. If a model gets **nothing correct**, relative rewards are all zero and **training loss stays zero** — start from a sufficiently capable model.
7. Small-sample evaluation has **high variance**; use larger token limits and the full test set in practice.
8. Full pipeline covered: build math **evaluation dataset** → design **reward function** → design **evaluation process** → run the **GRPO cycle** on an existing instruct model to improve its math capability.
