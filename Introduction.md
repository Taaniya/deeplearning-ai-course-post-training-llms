# Notes: Introduction

---

## 1. Two Phases of Training an LLM

| Aspect | Pre-training | Post-training |
|---|---|---|
| **Goal** | Model learns to predict the next word or token | Model is further trained to perform more specific tasks, such as answering questions |
| **Data scale** | Trillions or tens of trillions of tokens of text | Much smaller datasets |
| **Compute / cost** | The bulk of training compute and cost | Much faster and cheaper |
| **Time** | For very large models, can take months | Relatively quick |

**Key takeaway:** Pre-training builds the base model; post-training customizes it for specific tasks, and it is far cheaper and faster.

---

## 2. What the Course Covers

- Three common, newer ways to post-train and customize LLMs.
- You will **download a pre-trained model and post-train it yourself**, in a relatively computationally affordable way.
- The three techniques:
  1. Supervised Fine-Tuning (SFT)
  2. Direct Preference Optimization (DPO)
  3. Online Reinforcement Learning

---

## 3. Technique 1 — Supervised Fine-Tuning (SFT)

**Definition:** Trains a model on labeled prompt-response pairs.

**How it works:**
- The model learns to follow instructions or use tools by replicating the input prompt → response relationship.

**Strengths:**
- Especially effective for **introducing new behaviors** or making **major changes** to the model.

**Hands-on in the course:**
- Fine-tune a **small Qwen model** to follow instructions.

---

## 4. Technique 2 — Direct Preference Optimization (DPO)

**Definition:** Teaches a model by showing it both good and bad answers.

**How it works:**
- For the same prompt, the model is given two options, one preferred over the other.
- Through a **contrastive loss**, DPO pushes the model closer to good responses and away from bad responses.

**Example from the lecture (changing identity):**
- The model says, "I'm your assistant," but you want it to say, "I'm your AI assistant."
- Label **"I'm your assistant"** as the **bad** response.
- Label **"I'm your AI assistant"** as the **good** response.

**Hands-on in the course:**
- Use DPO on a **small Qwen instruct model** to change its identity.

---

## 5. Technique 3 — Online Reinforcement Learning

**Definition / process:**
1. Give the LLM prompts.
2. The LLM generates responses.
3. A **reward function** scores the quality of the answers.
4. The model is **updated based on these reward scores**.

### 5.1 Ways to Obtain Rewards

#### A. Reward model trained from human judgments
- Start with **human judgments** of the quality of responses.
- Train a function to assign scores to responses in a way that is **consistent with the human judgments**.
- The most common algorithm for this is probably **Proximal Policy Optimization (PPO)**.

#### B. Verifiable rewards
- Applies to tasks with **objective correctness measures**, such as **math** or **coding**.
- **Math:** use math checkers to verify solutions.
- **Coding:** use **unit tests** to objectively measure whether generated code is correct.
- This measure of correctness gives you the reward function.
- A powerful algorithm for using these reward functions is **Group Relative Policy Optimization (GRPO)**, introduced by **DeepSeek**.

**Hands-on in the course:**
- Use **GRPO** to train a **small Qwen model** to solve math problems.

---

## 6. Summary Comparison of the Three Techniques

| Technique | Training signal | Key idea | Course project |
|---|---|---|---|
| **SFT** | Labeled prompt-response pairs | Replicate the prompt → response relationship; good for new behaviors and major changes | Fine-tune a small Qwen model to follow instructions |
| **DPO** | Preferred vs. non-preferred response for the same prompt | Contrastive loss moves the model toward good and away from bad responses | Change the identity of a small Qwen instruct model |
| **Online RL** | Reward scores on the model's own generated responses | Model is updated based on reward scores (from a reward model or verifiable rewards) | Use GRPO to train a small Qwen model on math problems |

---

## 7. Key Terms and Acronyms

- **Pre-training** — learning to predict the next word/token on very large text corpora.
- **Post-training** — further training to perform more specific tasks, using smaller datasets.
- **SFT** — Supervised Fine-Tuning.
- **DPO** — Direct Preference Optimization.
- **Contrastive loss** — the loss used in DPO to push toward good and away from bad responses.
- **Online reinforcement learning** — the model generates responses, a reward function scores them, and the model is updated.
- **Reward function / reward model** — scores the quality of responses.
- **PPO** — Proximal Policy Optimization; the most common algorithm (probably) for training with a reward model learned from human judgments.
- **Verifiable rewards** — rewards from objective correctness checks (math checkers, unit tests).
- **GRPO** — Group Relative Policy Optimization; introduced by DeepSeek.

---

## 8. What's Next

- **Lesson 1** is an overview of post-training methods, covering:
  - **When** you should do post-training.
  - The **menu of post-training options** you can choose from.

---

## 9. Acknowledgments (as mentioned in the lecture)

- Oleksii Kuchaiev (Nvidia)
- Jiantao Jiao (UC Berkeley)
- Esmaeil Gargari (DeepLearning.AI)
