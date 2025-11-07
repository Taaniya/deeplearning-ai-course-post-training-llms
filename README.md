 # Course Notes 

**Course goals -**
* Learn to post train and customize LLMs in 3 new ways – SFT, DPO, Online RL
* Download a pre-trained model and post train on jupyter notebook
* 3 ways to customize LLMs -
    * Learn – SFT – Supervised fine-tuning
    * Learn – DPO – Direct preference optimization
	
**Online reinforcement learning -**
Supervised Fine Tuning -
* SFT – trains the model on labelled prompt-response pairs in the hope to learn to follow instructions or use tools by replicating that input prompt into desired response relationship
* SFT is especially effective for introducing new behaviours or making major changes to the model
* In this course, we’ll fine tune the Qwen model to follow instructions

**Direct Preference Optimization –** 
* DPO teaches a model by showing it both good and bad answers
* DPO gives the model 2 options for the same prompt – 1 preferred over the other
* DPO through a constructive loss, pushes a model closer to good and away from bad responses
* We’ll use DPO on a small Qwen instruction model to change its identity

**Online RL –**
* Idea in online RL is about letting the model itself explore responses.
* In Online RL, we feed the LLM a prompt, it responds and then a reward function scores the quality of the response. The model then gets updated based on these reward scores.
    * One way to get a reward model to give reward scores is to start with human judgements of the quality of responses. Then we can train a function to assign scores to the responses in a way that’s consistent with human judgements. The most common algorithm for this is PPO – Proximal Policy Optimization
    * Another way to come up with rewards is via verifiable rewards which applies to tasks of objective correctness measures - like Math or coding.
    * We can use math for checkers or, for coding use unit tests to measure in an objective way
* If generated math solutions or code is actually correct. This measure of correctness then gives you a reward function.
* A powerful algorithm for using this reward function is GRPO – Group Relative Policy Optimization – introduced by DeepSeek
* In this course we’ll use GRPO on Qwen


## Lecture 1 – Introduction to post-training

* The objective during pre-training is to minimize the negative log probability for each token conditioned on the previous tokens.
* This way we’re training the model to predict the next token given the previous tokens
* Contrast to pre-training, during post-training the training method is that the model is only trained on the responses and not the prompts – the loss function computes loss considering only the tokens of the responses and not the prompt

## Lecture 2 - SFT Supervised Finetuning
* SFT works by minimizing the negative log likelihood of the response given the prompt, where the likelihood of the responses is the product of the probability of tokens in the response given prior tokens including the prompt
* This way we maximize the probability of outputting the expected responses given the prompt
* That’s why SFT is like teaching the model to imitate the responses to become an instruct model
* For data curation techniques, using K rejection sampling method, we can generate multiple responses by the model and choose the best using any automated method, reward function etc.

## Lecture 4 - DPO - Direct Preference Optimization
* DPO can be considered as a contrastive learning method from both positive and negative responses

## Lecture 6 - Online Reinforcement learning

* Based on the reward model output for LLM responses, the language model update can use different algorithms. In this course we’ll go over 2 of them –
1.	PPO – Proximal Policy Optimization
2.	GRPO – Grouped relative policy optimization

Different choices of reward function in ORL –
Trained model – The model is trained such that the preferred responses are given higher reward by the model and less preferred responses are given lower reward. The preferences are collected from human labelled data


#### PPO -
(With reference to architecture diagram for PPO and GRPO) -
* The policy model here is the LLM itself
* Yellow blocks refer to trainable models in this architecture where the model weights are updated.
* Blue blocks refer to frozen models with frozen and won’t undergo any weight update in this process of online RL
* Value model is a critic model that tries to assign credits to each individual token so that one can decompose the response level reward into a token level reward
* Essentially, after we get a reward, and the value model’s output, we use the technique GAE to estimate ‘advantage’ which characterizes the credits for each individual token / contribution of each token to the entire process.
* Thus, by looking at the individual advantage, we can use that as a signal to guide the update of the policy model
* So, in PPO, essentially, we try to maximize return or the advantage for your current policy model – pi_theta

* Since here, we cannot sample the most recent output from the policy model, we try to maximize the expected advantage basically. This expected advantage is denoted by A_t
* Direct Ratio – pi_theta / pi_theta_old
* Pi_theta = current step LLM
* Pi_theta_old = previous step LLM
* The clip function is to make sure that this direct ratio is not too large or too small during this training process

#### GRPO –
* This is very similar to PPO where it also uses advantage and maximizes the same formula that PPO does to update the model
* However, it differs in how it calculates the advantage
* In this case, the policy model generates multiple responses as a group. For each prompt, we have 2 responses generated.
* We still use the reference and reward models the KL divergence & reward for each response
* Then we get a group for same queries with multiple outputs, and multiple rewards
* A group computation is used to calculate the relative reward for each of the output and we assume that the relative reward will just be the relative advantage for each individual token. This way we get the more brute force estimation of advantage for each token and we use that advantage to update the policy model
* Now, everything after getting the advantage PPO and GRPO is the same. Only difference is how advantage itself is calculated



**GRPO vs PPO in summary –**
* PPO relies on actual value model that needs to be trained during the entire process whereas GRPO gets rid of this value mode and hence can be more memory efficient
* Though the cost of getting rid of such value model is that your advantage estimation can be more brute force & this advantage stays the same for every token in the same response
* Whereas for PPO, the advantage can be different for each token
* In short, what PPO does is to use actual value / critic model to assign credit for each individual token. In this way, in the entire generation , each word / token will have different advantage value which shows which token is more important and which is less.
* Whereas in GRPO, because we got rid of value model, each token will have the same advantage as long as they stay in the same output
* This way, PPO usually gives more fine grained advantage feedback for each individual token 
* While GRPO gives more uniform advantage for tokens in the same response 

GRPO – 
1.	Well suited for binary (often correctness based) reward
2.	Requires larger amount of samples (since it only assigns credits to full responses rather than individual tokens)
3.	Requires less GPU memory (since no value model needed)

PPO –
-	Works well with reward model or binary reward
-	More sample efficient with a well-trained value model
-	Requires more GPU memory (since value model is used)

**References -**
* DeepLearning.AI course - https://learn.deeplearning.ai/courses/post-training-of-llms/lesson/n25bz/introduction
* [vLLM - A fast and efficient library for LLM inference and serving](https://docs.vllm.ai/en/latest/)
* [sglang - A high performance framework for LLM serving](https://docs.sglang.ai/)
* [Nvidia TensorRT](https://developer.nvidia.com/tensorrt-getting-started)
* [DPO paper - Your LM is secretly a reward model](https://openreview.net/pdf?id=HPuSIXJaa9)
* [PPO paper](https://arxiv.org/pdf/1707.06347)
* [PPO HF article](https://huggingface.co/blog/deep-rl-ppo)
* [GRPO paper](https://arxiv.org/pdf/2402.03300)
* [GAE - Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)
* [QWen instruct HF model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
* [Qwen2.5 Technical report](https://arxiv.org/pdf/2412.15115)



