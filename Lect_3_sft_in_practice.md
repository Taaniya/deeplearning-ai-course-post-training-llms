# Lecture Notes — SFT in Practice (with code example)

---

## 1. Lecture Goal

- Build an **SFT (supervised fine-tuning) pipeline** on a small-scale training dataset.
- Start from a **base language model**, prepare labeled chat / instruction-following data, run SFT, and obtain a fine-tuned model that can chat with a user.

---

## 2. Concept Recap: What SFT Is

- SFT is for **imitating example responses**.
- Starting point: any language model, which can be a **base model**. A base model only tries to **predict the next most probable tokens** based on the user query.
- Curate **chat data or instruction-following data** where the assistant responds to user queries in a more natural fashion.
  - Example: user asks "How are you?" → ideal response "I'm doing great."
- Use this **labeled data** for supervised fine-tuning on top of the base model → a **fine-tuned model** that chats more fluently.

---

## 3. Libraries Used

| Library | Purpose |
|---|---|
| `torch` | Essential for training with PyTorch |
| `pandas` | Displaying tables of the dataset |
| HuggingFace `datasets` | Loading datasets; `Dataset` class for defining them |
| HuggingFace `transformers` | Training arguments, `AutoTokenizer`, `AutoModelForCausalLM` |
| HuggingFace `TRL` | `SFTTrainer`, data collator, `SFTConfig` for setting up the SFT process |

---

## 4. Helper Functions

### 4.1 `generate_responses`
- **Arguments:** model, tokenizer, user message, optional system message, maximum number of new tokens.
- **Steps:**
  1. Start with a clean empty list of messages.
  2. If a system message exists, append a dictionary with role `system` and the system message as content.
  3. Append the user message in the same way (role `user`).
  4. Use the tokenizer's **chat template** function to convert messages into the format the model was trained on.
     - For Qwen 3 specifically, `enable_thinking` must be set to **False** so the model does not enter thinking mode.
  5. Call the tokenizer to convert the text prompt into **tokens**, and send them to the **same device as the model** (e.g., GPU).
  6. Call the model's `generate` with `max_new_tokens` set to the passed argument, to control how many new tokens are generated.
  7. Extract the generated IDs (still tokens), then call `tokenizer.decode` to turn them into a **text response**, and return it.
- **Tip from the lecture:** Besides `model.generate`, try **vLLM, SGLang, or TensorRT** — inference libraries that can be faster and more efficient than HuggingFace's `model.generate`.

### 4.2 `test_model_with_questions`
- **Arguments:** model, tokenizer, list of questions, optional system message, and a title for printing.
- Prints the title, calls `generate_responses` for each question, and prints the model input and model output.

### 4.3 `load_model_and_tokenizer`
- **Arguments:** model name from HuggingFace, and whether to use GPU.
- Loads the tokenizer with `AutoTokenizer` and the model with `AutoModelForCausalLM`.
- If using GPU, sends the model to CUDA (assuming an NVIDIA GPU).
- **Fallback chat template:** if the tokenizer has no chat template, one is created manually.
  - It iterates over all messages: for role `system`, builds a string with "system" followed by the content; for `user`, "user" followed by the content; for `assistant`, "assistant" followed by the content.
- **Tokenizer config:** if no pad token exists, it defaults to the **end-of-sequence token**.
- Returns the loaded model and tokenizer.

### 4.4 `display_dataset`
- Displays the dataset in a **Jupyter-notebook-friendly** way.
- Goes through the examples, takes the user message and the system message, turns them into rows of a table, and displays it with pandas.

---

## 5. Testing the Base Model (Before SFT)

- Setup: `use_gpu = False` (the DeepLearning.AI platform only provides CPU in this environment); set it to `True` on your own GPU machine.
- Test questions:
  1. Give me a one-sentence introduction of a language model.
  2. Calculate 1 + 1 − 1.
  3. What is the difference between a thread and a process?
- Model used: small Qwen 3 model, **Qwen3-0.6B-Base** (no SFT applied).
- **Observation:** the base model outputs **random tokens** for the given instructions.
- **Two reasons given:**
  1. The **chat template** used was **never seen during pre-training**.
  2. A pre-training (base) model is **not great at answering user questions**.

---

## 6. Testing an SFT Checkpoint (After SFT)

- A checkpoint trained through SFT on the base model (training detailed later in the lecture) was loaded and tested on the same questions.
- **Observation:** the output is **much more natural**; the model responds properly to all requests — one-sentence introduction of a language model, the calculation, and explaining thread vs. process.
- The lecturer trained the Qwen 3 model with SFT to enable a before/after comparison.

---

## 7. Running SFT at Small Scale

### 7.1 Why scale down
- Due to **resource limitations**, SFT is not performed on Qwen3-0.6B here.
- Instead: a **much smaller model** and a **much smaller dataset**.
- Learners are encouraged to use the entire dataset on the same model (Qwen3-0.6B) on their own GPU to reproduce the lecturer's SFT results.

### 7.2 Model
- `HuggingFaceTB/SmolLM2-135M` — a **135 million parameter** model, smaller than Qwen3-0.6B.
- On your own GPU, change the model name to Qwen 3.

### 7.3 Dataset
- A training dataset of **prompt–response pairs** created beforehand (user prompts and assistant responses).
- Instructions are **diverse**: questions, instructions, translation requests, etc.
- Without a GPU, train only on the **first 100 samples** for illustration; with a GPU, train on the entire dataset to recover Qwen 3 performance.

### 7.4 SFT Trainer Configuration (key hyperparameters)

| Hyperparameter | Explanation from the lecture |
|---|---|
| **Learning rate** | Learning rate for training. You usually need to experiment a lot to find the best value for your own dataset and model. |
| **Number of training epochs** | Set to **1** here to speed things up. Set to 2 or higher to train on the dataset multiple times. |
| **Per-device train batch size** | Batch size on each device/GPU. |
| **Gradient accumulation steps** | Number of steps before performing a gradient update. |
| **Gradient checkpointing** | When enabled, helps **reduce GPU memory** by skipping storage of some activations. Set to **False** here. If you hit out-of-memory errors, setting it to **True** is one of the first things to try. |
| **Logging steps** | Frequency of logging the training process; affects the outputs seen during training. |

### 7.5 Effective (Total) Batch Size
- **Effective batch size = per-device train batch size × number of GPUs × gradient accumulation steps.**
- Example with 8 GPUs and per-device batch size 2: without accumulation → 2 × 8 = **16**.
- Gradient accumulation steps multiply on top of that.
- In this lab: 1 device × per-device batch size 1 × gradient accumulation 8 = **effective batch size of 8**.
- **Trade-off:** a larger per-device batch size usually needs **more memory on each GPU**. Gradient accumulation **effectively increases batch size without increasing memory usage**.

### 7.6 Launching Training
- Create an `SFTTrainer` with:
  - the **model**,
  - the **SFTConfig** as arguments,
  - the prepared **training dataset**,
  - the **tokenizer** as the processing class.
- Call train. A **progress bar** shows training progress (1 epoch).
- Step count: 100 samples ÷ batch size 8 → **13 gradient descent steps**.
- Training the small model on 100 samples takes **on the scale of minutes**.

---

## 8. Results After Small-Scale SFT

- Tested by passing the SFT-trained model to the test function with the prepared questions.
- The model gives **reasonable responses**, but:
  - sometimes **repetitive**,
  - sometimes **gives the wrong answer**.
- **Reasons given:**
  1. The model is **small**.
  2. The dataset has only **100 samples**, which may not be enough to update the model to a good shape.
- This was done due to limited resources; learners are encouraged to train **Qwen3-0.6B on the full dataset** on their own GPU to reproduce the earlier illustrated results.

---

## 9. Key Takeaways

1. A **base model** predicts next tokens and, without SFT, produces **random-looking output** for chat-style instructions — partly because the **chat template was unseen in pre-training**.
2. **SFT on prompt–response data** turns a base model into an **instruct model** that chats naturally (demonstrated with Qwen3-0.6B-Base).
3. Core SFT workflow with HuggingFace: load model + tokenizer → prepare prompt–response dataset → configure `SFTConfig` → run `SFTTrainer` → test the model.
4. Important hyperparameters: **learning rate** (needs tuning), **epochs**, **per-device batch size**, **gradient accumulation steps**, **gradient checkpointing**, **logging steps**.
5. **Effective batch size = per-device batch size × #GPUs × gradient accumulation steps**; gradient accumulation raises effective batch size without extra memory.
6. **Gradient checkpointing** reduces GPU memory use and is a first thing to try for out-of-memory problems.
7. A tiny model (135M) with only 100 samples gives limited quality (repetition, wrong answers) — scale model and data up on your own GPU for good results.
8. Faster inference alternatives to `model.generate`: **vLLM, SGLang, TensorRT**.

---

## 10. What's Next

- The next lesson covers the **basics of DPO**.
