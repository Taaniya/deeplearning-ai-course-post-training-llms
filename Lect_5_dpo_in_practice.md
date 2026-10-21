# Lecture notes: DPO in Practice


---

## 1. Lecture Overview

- The lesson is about **building a DPO pipeline on a small-scale training dataset**, done through hands-on coding.
- Reminder: **DPO is a contrastive learning method** that learns from **both positive and negative samples**.
- The lesson covers:
  1. Data curation for DPO
  2. Running the full DPO training cycle on a smaller model
  3. Comparing the model's identity output **before and after** DPO training

---

## 2. The Goal of the Lab: Changing a Model's Identity

| Item | Description |
|---|---|
| Starting model | A small **Qwen instruct model**, which has its own identity as "Qwen" |
| Original behavior | When asked "Who are you?", it answers "I'm Qwen" |
| Target behavior | After DPO, it should answer "I'm **Deep Qwen**" |
| Positive sample | Response using the name **Deep Qwen** |
| Negative sample | Response using the name **Qwen** |

- Comparison data is created by changing the identity name from Qwen to Deep Qwen.
- DPO is run **on top of an existing instruct model**.
- Result: a fine-tuned Qwen model with a new identity.

---

## 3. Setup and Libraries

- Libraries imported for the DPO coding part:
  - `torch`, `pandas`
  - `transformers` — `AutoTokenizer`, `AutoModelForCausalLM`
  - `TRL` — the new **`DPOTrainer`** and **`DPOConfig`**
  - `datasets` — `load_dataset` and the dataset type
- Helper functions (implemented in the previous lesson): **generate responses**, **test model with questions**, **load model and tokenizer**.
- **GPU setting:** set to `False` since the lab mostly operates on CPU machines. On your own machine, feel free to set it to `True`.

---

## 4. Testing the Original Instruct Model

- Model loaded: **Qwen 2.5-0.5B Instruct**.
- Test questions about identity:
  - "What is your name?"
  - "Are you ChatGPT?"
  - "Tell me about your name and organization."
- Observed outputs:
  - Replies "I'm Qwen, a language model trained by Alibaba Cloud."
  - Replies "I'm Qwen" even when asked "Are you ChatGPT?"
- **Conclusion:** the model has a clear identity of Qwen and knows it was created by Alibaba Cloud.

---

## 5. Result of a Pre-trained DPO Model (Demo)

- A trained model, **Qwen 2.5-0.5B DPO**, is tested on the same questions.
- Training data was curated to change the identity from Qwen to Deep Qwen by **adding "Deep Qwen" in most of the responses**.
- After DPO post-training, the model generates **"Deep Qwen"** as its identity across multiple responses.

---

## 6. The Full DPO Procedure (Walkthrough)

- The walkthrough uses a **HuggingFace small LLM** (slightly smaller than Qwen 2.5) so it can run without GPUs.
- If you have your own GPU, start from **Qwen 2.5** to reproduce the exact results shown in the lecture.

### 6.1 Step 1 — Load the Identity Dataset

- Start from an **identity dataset from HuggingFace**, containing prompts and responses about identity-related questions.
- Example: user asks "Who are you?" → assistant replies "I'm an assistant, a helpful AI created by a developer…"
- It may also include **multi-round conversations** about identity and the model's developer.
- This gives a set of prompts that query the model about its own identity.

### 6.2 Step 2 — Set Parameters

- Set parameters to change the name from **Qwen → Deep Qwen**.
- Define a **new system prompt** to replace the original Qwen 2.5 system prompt.
  - Reason: the original Qwen 2.5 system prompt **already contains its own identity and developer**.
  - The new system prompt is simply a helpful assistant, so the model isn't told about its identity or developer.
- **CPU-only shortcut:** use only the **first 5 samples** of the original dataset for illustration and to avoid long waits.

### 6.3 Step 3 — Build the DPO Dataset

A DPO dataset requires a **preferred** and a **less preferred** answer, called **`chosen`** and **`rejected`**.

**Procedure in the dataset-building function:**
1. Start from existing conversations in the identity dataset.
2. **Extract the last prompt from "human"** as the prompt.
3. **Generate a response** to that prompt using the current model.
   - If generation fails, double-check and print the potential error.
4. **Rejected response** = the model's **own generated response**
   - Reason: the goal is to change the model's own identity.
5. **Chosen response** = the model's own response with the original name **Qwen replaced by Deep Qwen**.

**Structure of each example:**

| Field | Contents |
|---|---|
| Chosen | System prompt + original sample prompt + chosen response (Qwen → Deep Qwen) |
| Rejected | System prompt + original sample prompt + the original model's own response |

- Result: preferred responses as `chosen`, less preferred responses as `rejected`.

### 6.4 Step 4 — Map the Function to the Raw Dataset

- Map the DPO chat-ML building function to the raw dataset and **remove unnecessary columns**.
- On CPU only the **5 samples** are mapped.
- The function uses the model to generate rejected responses, which **takes time**.
  - The original full-size dataset has **1000 samples**, so generation would take much longer.
- A **fully mapped dataset** is provided that converts Qwen's own responses into the Deep Qwen identity.
- In the mapped data:
  - **Chosen** always answers with **Deep Qwen** as its identity.
  - **Rejected** always has **Qwen**.
  - **This is the only difference** among all conversations in the DPO dataset.

---

## 7. DPO Training

### 7.1 Data Size

- Without GPU, take only the **first 100 samples** to speed up the process.

### 7.2 DPO Config

- Similar to the SFT config: 
  - per-device train batch size
  - gradient accumulation steps
  - number of training epochs
  - learning rate
  - logging steps
- **One new hyperparameter: `beta`**
  - It appears in the original DPO formula.
  - It is a hyperparameter that decides **how important the log differences are**.
  - It is an **important hyperparameter to tune together with the learning rate** for the best DPO performance.

### 7.3 Trainer Setup (`DPOTrainer`)

| Argument | Setting |
|---|---|
| `model` | The model loaded earlier |
| Reference model | Usually set to `None` → automatically **creates a copy of the original model as the reference model and freezes its weights** |
| `args` | The DPO config set before |
| `processing_class` | The tokenizer |
| `train_dataset` | The previously built DPO dataset |

### 7.4 Training Run

- 100 samples, **1 epoch**, **batch size 8** → a certain number of steps to finish the DPO process.
- Because it's a **smaller model with a smaller dataset**, this training is **not expected to have the same effect** as the earlier shown results.

---

## 8. Results of Full-Scale Training

- A code snippet shows the result of a **complete training on Qwen 2.5-0.5B** on the same dataset at full scale.
- After training:
  - The model's **identity changes to Deep Qwen**.
  - **The rest does not change**, including its **developer** and its **own knowledge**.
- You can swap in the fully trained Qwen to see the results versus the small-scale run.
- The small-scale DPO (very small dataset) was used to **speed up training** and allow seeing the full DPO training without long waits given **limited computational resources**.

---

## 9. Additional Notes from the Transcript's Tail (Repeated Segment)

- After loading the identity dataset, you get a set of prompts asking about the model's own identity.
- The system prompt replaces the original Qwen 2.5 one, which contains its name and identity; the new one says only that it's a helpful assistant, so the model isn't told about its identity or developer.
- The dataset size is restricted to the first 5 samples for illustration.

---

## 10. Key Takeaways

1. **DPO** is contrastive: it learns from **chosen (positive)** and **rejected (negative)** samples.
2. DPO data can be built by using the **model's own response as rejected** and a **modified version of it as chosen** (here: Qwen → Deep Qwen).
3. A **new system prompt** is needed so the original identity/developer info doesn't leak into the data.
4. `DPOConfig` mirrors `SFTConfig`, plus the extra **`beta`** hyperparameter (tune it with the learning rate).
5. In `DPOTrainer`, setting the reference model to `None` makes it **copy and freeze** the original model automatically.
6. Full-scale DPO changed **only the identity**; developer info and other knowledge stayed intact.
7. The small-scale CPU run is illustrative only and not expected to match full-scale results.

---

## 11. Up Next

- The next lesson covers the **basics of online reinforcement learning**.
