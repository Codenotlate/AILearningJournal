# Week 3 Notes

- [Week 3 Notes](#week-3-notes)
  - [Key questions on this week](#key-questions-on-this-week)
  - [RLHF - Aligning models with human values](#rlhf---aligning-models-with-human-values)
  - [Lab3 Notes (https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/main/Week-3/Lab\_3\_fine\_tune\_model\_to\_detoxify\_summaries.ipynb)](#lab3-notes-httpsgithubcomryota-kawamuragenerative-ai-with-llmsblobmainweek-3lab_3_fine_tune_model_to_detoxify_summariesipynb)
  - [For the Generative AI project lifecycle, we are now looking at the Application integration stage](#for-the-generative-ai-project-lifecycle-we-are-now-looking-at-the-application-integration-stage)
  - [Model optimizations for deployment](#model-optimizations-for-deployment)
  - [Generative AI Project Lifecycle Cheat Sheet](#generative-ai-project-lifecycle-cheat-sheet)
  - [Other challenges with LLM outside training part](#other-challenges-with-llm-outside-training-part)
  - [Connect to external data sources and applications to solve above challenges](#connect-to-external-data-sources-and-applications-to-solve-above-challenges)
  - [How to connect LLM to external data sources - Retrieval Augmented Generation (RAG)](#how-to-connect-llm-to-external-data-sources---retrieval-augmented-generation-rag)
  - [How LLM interact with external applications](#how-llm-interact-with-external-applications)
  - [Chain-of-thought (help with complex reasoning)](#chain-of-thought-help-with-complex-reasoning)
  - [Program-aided language models (PAL)](#program-aided-language-models-pal)
  - [ReAct: Combining reasoning and action](#react-combining-reasoning-and-action)
  - [LLM application architectures](#llm-application-architectures)
  - [Week3 Resources: https://www.coursera.org/learn/generative-ai-with-llms/supplement/89tR9/week-3-resources](#week3-resources-httpswwwcourseraorglearngenerative-ai-with-llmssupplement89tr9week-3-resources)
  - [responsible AI](#responsible-ai)
  - [other ongoing research areas](#other-ongoing-research-areas)


## Key questions on this week
- What is RLHF (reinforcement learning from human feedback)? What problem does it solve?
- How to use LLM as a reasoning engine to generate subtasks and take actions on those tasks?
- What is RAG? How to have LLM getting external data and utilize those for tasks?
- Discussions on responsible AI, like the safety issue.
## RLHF - Aligning models with human values
- Models can behave badly given they are trained on massive amount of internet data which may contain harmful content.
- We want answers from LLM to follow HHH principle - Helpful, Honest, Harmless. RLHF can help with reducing the toxicity and incorrect information.
    
    ![image.png](Week%203%20Notes/image.png)
    
    One potentially exciting application of RLHF is the personalizations of LLMs, where models learn the preferences of each individual user through a continuous feedback process. This could lead to exciting new technologies like individualized learning plans or personalized AI assistants. 
    
- High level review of RL concepts
    
    ![image.png](Week%203%20Notes/image%201.png)
    
    - Reinforcement learning is a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward.
    
    ![image.png](Week%203%20Notes/image%202.png)
    
    - In this framework, the agent continually learns from its experiences by taking actions, observing the resulting changes in the environment, and receiving rewards or penalties, based on the outcomes of its actions. By iterating through this process, the agent gradually refines its strategy or policy to make better decisions and increase its chances of success.
- How RL is used for fine-tune LLMs
    
    ![image.png](Week%203%20Notes/image%203.png)
    
    - the agent's policy that guides the actions is the LLM, and its objective is to generate text that is perceived as being aligned with the human preferences.
    - The environment is the context window of the model, the space in which text can be entered via a prompt.
    - The state that the model considers before taking an action is the current context. That means any text currently contained in the context window.
    - The action here is the act of generating text. This could be a single word, a sentence, or a longer form text, depending on the task specified by the user. The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion.
        - At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space.
    - The reward is assigned based on how closely the completions align with human preferences.  Determining the reward is more complicated in LLM case comparing to other regular RL cases.
        - One way you can do this is to have a human evaluate all of the completions of the model against some alignment metric, such as determining whether the generated text is toxic or non-toxic. This feedback can be represented as a scalar value, either a zero or a one. The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier, enabling the model to generate non-toxic completions.
        - Another less costly and time efficient way is use an additional model, known as the reward model, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences.
            - You'll start with a smaller number of human examples to train the secondary model by your traditional supervised learning methods.
            - Once trained, you'll use the reward model to assess the output of the LLM and assign a reward value, which in turn gets used to update the weights off the LLM and train a new human aligned version.
    - In the context of language modeling, the sequence of actions and states is called a rollout, instead of the term playout that's used in classic reinforcement learning.
- RLHF step1 - Get feedback from humans to prepare the training dataset for reward model
    - The first step in fine-tuning an LLM with RLHF is to select a model to work with and use it to prepare a data set for human feedback.
        - The model you choose should have some capability to carry out the task you are interested in, whether this is text summarization, question answering or something else. In general, you may find it easier to start with an instruct model that has already been fine tuned across many tasks and has some general capabilities.
        - You'll then use this LLM along with a prompt data set to generate a number of different responses for each prompt.
            
            ![image.png](Week%203%20Notes/image%204.png)
            
    - The next step is to collect feedback from human labelers on the completions generated by the LLM.
        
        ![image.png](Week%203%20Notes/image%205.png)
        
        - The task for your labelers is to rank the three completions in order of helpfulness from the most helpful to least helpful.
        - This process then gets repeated for many prompt completion sets, building up a data set that can be used to train the reward model that will ultimately carry out this work instead of the humans.
        - The same prompt completion sets are usually assigned to multiple human labelers to establish consensus and minimize the impact of poor labelers in the group.
        - The clarity of your instructions can make a big difference on the quality of the human feedback you obtain.
            - Labeler instruction example
                
                ![image.png](Week%203%20Notes/image%206.png)
                
                - In general, the more detailed you make these instructions, the higher the likelihood that the labelers will understand the task they have to carry out and complete it exactly as you wish.
    - Before using the feedback data to train the reward model, you need to convert the ranking data into a pairwise comparison of completions. In other words, all possible pairs of completions from the available choices to a prompt should be classified as 0 or 1 score.
        
        ![image.png](Week%203%20Notes/image%207.png)
        
        - For each pair, you will assign a reward of 1 for the preferred response and a reward of 0 for the less preferred response.
        - Then you'll reorder the prompts so that the preferred option comes first. This is an important step because the reward model expects the preferred completion, which is referred to as Yj first.
        - Note that while thumbs-up, thumbs-down feedback is often easier to gather than ranking feedback, ranked feedback gives you more prom completion data to train your reward model. As you can see, here you get three prompt completion pairs from each human ranking.
- RLHF step2 - train the reward model
    - The reward model is usually also a language model -  trained using supervised learning methods on the pairwise comparison data that you prepared from the human labelers assessment off the prompts.
        
        ![image.png](Week%203%20Notes/image%208.png)
        
        - For a given prompt X, the reward model learns to favor the human-preferred completion y_ j, while minimizing the loss (Self correction: The loss function in above image is incorrect, it should be loss = -log(sigmoid(rj-rk). As we want to minimize the loss which equals to maximize the log sigmoid of the reward difference, r_j-r_k.)
            
            ![image.png](Week%203%20Notes/image%209.png)
            
    - After the reward model is trained, it can be used as a binary classifier.
        
        ![image.png](Week%203%20Notes/image%2010.png)
        
        - Logits are the unnormalized model outputs before applying any activation function.
        - If you apply a Softmax function to the logits, you will get the probabilities.
- RLHF step3 - Use reward model in the whole RLHF process
    - One single iteration of the RLHF process
        
        ![image.png](Week%203%20Notes/image%2011.png)
        
        - First, you'll pass a prompt from your prompt dataset. In this case, a dog is, to the instruct LLM, which then generates a completion, in this case a furry animal.
        - Next, you sent this completion, and the original prompt to the reward model as the prompt completion pair. The reward model evaluates the pair based on the human feedback it was trained on, and returns a reward value. A higher value such as 0.24 as shown here represents a more aligned response. A less aligned response would receive a lower value, such as negative 0.53.
        - You'll then pass this reward value for the prom completion pair to the reinforcement learning algorithm to update the weights of the LLM, and move it towards generating more aligned, higher reward responses.
        
        ![image.png](Week%203%20Notes/image%2012.png)
        
    - These iterations continue for a given number of epics, similar to other types of fine tuning. Here you can see that the completion generated by the RL updated LLM receives a higher reward score, indicating that the updates to weights have resulted in a more aligned completion.
        
        ![image.png](Week%203%20Notes/image%2013.png)
        
        ![image.png](Week%203%20Notes/image%2014.png)
        
        - You will continue this iterative process until your model is aligned based on some evaluation criteria. For example, reaching a threshold value for the helpfulness you defined. You can also define a maximum number of steps, for example, 20,000 as the stopping criteria.
        
        ![image.png](Week%203%20Notes/image%2015.png)
        
    - RL algorithm (a popular choice is proximal policy optimization (PPO))
        
        ![image.png](Week%203%20Notes/image%2016.png)
        
        - This is the algorithm that takes the output of the reward model and uses it to update the LLM model weights so that the reward score increases over time. There are several different algorithms that you can use for this part of the RLHF process. A popular choice is proximal policy optimization.
        - you don't have to be familiar with all of the details to be able to make use of it. However, it can be a tricky algorithm to implement and understanding its inner workings in more detail can help you troubleshoot if you're having problems getting it to work.
    - Proximal policy optimization (PPO) - chatGPT notes based on the video transcript
        - About the overall process
            
            ---
            
            ### 🧠 **Big Picture**
            
            - **Goal of PPO in RLHF**: Improve an LLM (like GPT) so its outputs better align with **human preferences**.
            - **How**: Use reinforcement learning to gently steer the model’s behavior, while making sure updates are not too drastic (this is the “Proximal” part).
            
            ---
            
            ### 🪜 Step-by-Step Explanation
            
            ### **Step 1: Start with a Base LLM**
            
            - You begin with an already-trained LLM, like an instruct model.
            - This model can already generate completions to prompts.
            
            ---
            
            ### **Step 2: Phase I – Gather Data**
            
            - The LLM generates completions for a set of prompts.
            - These completions are then **scored by a reward model**.
                - This reward model mimics human judgment — it scores how helpful, honest, harmless, etc. the completions are.
            
            ---
            
            ### **Step 3: Estimate Value with a Value Function**
            
            - As the LLM generates each token in a completion, we estimate the **future total reward** for continuing the generation from this point — this is done by a special part of the model called the **value function**.
            - This value helps in understanding if a certain token choice is "promising."
            
            ✅ **Why it's important**: We want to guide training toward tokens and completions that lead to higher rewards.
            
            ---
            
            ### **Step 4: Compute Value Loss**
            
            - Value loss = difference between the predicted value and the actual reward.
            - We train the value function to get better at estimating future rewards.
            
            ---
            
            ### **Step 5: Phase II – Update the LLM (Policy Update)**
            
            - Here comes the reinforcement learning part.
            - We want to **increase the likelihood of generating better tokens** (those with high advantage).
            
            ### 🔍 PPO’s Policy Objective:
            
            It looks at:
            
            ```
            r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
            
            ```
            
            This ratio compares:
            
            - **π_θ**: Probability of action (token) under the new model
            - **π_θ_old**: Same but under the original model
            
            We **multiply this ratio by the advantage** (how much better or worse a token is than average):
            
            ```
            r_t(θ) * Â_t
            
            ```
            
            But — **to avoid huge updates**, we clip this ratio:
            
            ```
            min(r_t(θ) * Â_t, clip(r_t(θ), 1 - ε, 1 + ε) * Â_t)
            
            ```
            
            This “clip” ensures the new model doesn’t deviate too far from the old one. It enforces the **trust region**.
            
            ---
            
            ### **Step 6: Add Entropy Bonus**
            
            - We want the model to remain **creative** and explore different completions.
            - Entropy encourages diversity by preventing the model from becoming too deterministic.
            
            ---
            
            ### **Step 7: Compute Total PPO Objective**
            
            All losses are combined:
            
            ```
            PPO Loss = Policy Loss + C1 * Value Loss - C2 * Entropy Bonus
            
            ```
            
            (C1 and C2 are tunable hyperparameters.)
            
            ---
            
            ### **Step 8: Backpropagation & Update**
            
            - The total PPO loss is used to update the LLM’s weights.
            - After the update, we start a new cycle using the updated model.
            
            ---
            
            ### 🌀 This Process Repeats
            
            - Run a new round of data collection → reward → value estimation → PPO update.
            
            ---
            
            ### 💡 Intuition Behind It All
            
            - If a token leads to good outcomes (high reward), increase its probability.
            - But don’t change things too much at once.
            - Keep the model flexible and creative.
            
            ---
            
        - About ‘advantage’ part used in step5
            
            ---
            
            ### 🧠 What is the *advantage*?
            
            The **advantage** tells us:
            
            > How much better (or worse) an action was compared to what the model expected.
            > 
            
            It’s like asking:
            
            > “Was choosing this action at this moment a good idea?”
            > 
            
            ---
            
            ### 📦 In PPO (and most policy gradient methods), advantage is:
            
            ```
            Advantage = Reward - Baseline
            
            ```
            
            - **Reward**: What the model got from the environment (e.g. from the reward model in RLHF).
            - **Baseline**: What the model *expected* to get (often from a value function or critic (self added: value loss from PPO phase 1)).
            
            ---
            
            ### 🏠 Analogy: House Hunting
            
            Imagine you're choosing houses to buy (actions), and you expect that on average, you’ll get a $500k value (the baseline).
            
            If a house gives you:
            
            - $600k value → Advantage = +100k (good!)
            - $500k value → Advantage = 0 (as expected)
            - $400k value → Advantage = -100k (bad!)
            
            So, advantage is just a way to **center rewards** around expectations, and figure out how surprised (positively or negatively) we should be.
            
            ---
            
            ### ✅ Why Use Advantage in PPO?
            
            Because:
            
            - If the advantage is **positive**, we want to **increase** the chance of that action (we got more reward than expected).
            - If it's **negative**, we want to **decrease** the chance (it wasn't worth it).
            
            By using advantage:
            
            - PPO adjusts each action’s probability **based on how good it really was**, not just the raw reward.
            - It removes **baseline bias** so that the learning signal is clearer and less noisy.
            
            ---
            
            ### 🛠️ In Practice: How it works in PPO step 5
            
            Let’s say:
            
            - Old model said “Paris” with 0.6 probability.
            - New model boosts it to 0.9.
            - Advantage = +2.0 (means this was much better than expected)
            
            So the loss tries to **reinforce** this change — but carefully (via clipping).
            
            If instead advantage = -1.5, PPO says:
            
            > “Oh, we thought this action would be good, but it wasn’t — let’s reduce its chance next time.”
            > 
            
            ---
            
        - About step5 detailed example
            
            ---
            
            ### 🎯 Goal of Step 5
            
            We want the model to:
            
            - **increase** the chances of good actions (tokens),
            - **decrease** the chances of bad ones,
            - **but not change too much at once.**
            
            So Step 5 is where PPO says: *“Let’s improve, but carefully.”*
            
            ---
            
            ### 🤖 Simple Example
            
            Let’s say the prompt is:
            
            > “What’s the capital of France?”
            > 
            
            The current model generates:
            
            > “Paris”
            > 
            
            ✅ This is good — the reward model gives it a **high score**.
            
            Now suppose earlier the model was:
            
            - 60% confident to output “Paris”
            - 30% confident for “Lyon”
            - 10% for “Berlin”
            
            After PPO training, we **want the new model to increase the probability of “Paris.”**
            
            ---
            
            ### 🔍 Now the PPO Part
            
            PPO looks at how much the model changed its probability for each token using a ratio:
            
            ```
            r = π_new(token) / π_old(token)
            
            ```
            
            For “Paris”:
            
            - Old prob = 0.60
            - New prob = 0.90
            - So r = 0.90 / 0.60 = 1.5 ✅
            
            If we just multiply this with the advantage and say "Great! Let's go!", that would **make the model too confident** about “Paris” and maybe destroy the balance with other examples.
            
            ---
            
            ### 🧠 So Why Minimize PPO Loss?
            
            We define the **policy loss** like this:
            
            ```
            Loss = - min(r * advantage, clip(r, 1 - ε, 1 + ε) * advantage)
            
            ```
            
            Let’s break it down:
            
            ### If r is too big (>1 + ε):
            
            ➡️ Clipping kicks in and stops the model from changing too much.
            
            ➡️ This **prevents overfitting** and keeps the model stable.
            
            ### If r is < 1 and the action was **bad** (negative advantage):
            
            ➡️ The model reduces the probability of that token — good!
            
            ---
            
            ### ✅ Intuition: What Are We Minimizing?
            
            By minimizing this loss, we are:
            
            - **Encouraging the model to boost good tokens** ("Paris")
            - **Discouraging bad tokens** ("Lyon", "Berlin")
            - **Avoiding large jumps** in the probabilities — staying *proximal* to the original model
            
            PPO keeps learning **stable** and **gentle**.
            
            ---
            
            ### 🪄 TL;DR Analogy
            
            Imagine you're training a dog 🐶.
            
            - If it does something good ("sit"), you want to **reward it more** so it does it again.
            - But you don’t want to give **too many treats** (overfitting).
            - PPO is like giving just **enough treats**, and adjusting your tone slightly, without going overboard.
            
            ---
            
        
- RLHF potential issue - reward hacking & solution
    - An interesting problem that can emerge in reinforcement learning is known as reward hacking, where the agent learns to cheat the system by favoring actions that maximize the reward received even if those actions don't align well with the original objective.
        - In the context of LLMs, reward hacking can manifest as the addition of words or phrases to completions that result in high scores for the metric being aligned. But that reduce the overall quality of the language.
            
            ![image.png](Week%203%20Notes/image%2017.png)
            
            This language sounds very exaggerated.
            
            ![image.png](Week%203%20Notes/image%2018.png)
            
            The model could also start generating nonsensical, grammatically incorrect text that just happens to maximize the rewards in a similar way, outputs like this are definitely not very useful. 
            
    - To solve this - use initial LLM as performance reference, use KL divergence to compare and penalize the RL updated model if it shifts too much from the initial reference model.
        - KL divergence related
            - To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.
            - A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is TRL (Transformer Reinforcement Learning). In [this link](https://huggingface.co/blog/trl-peft) you can read more about this library, and its integration with PEFT (Parameter-Efficient Fine-Tuning) methods, such as LoRA (Low-Rank Adaption).
        - Case1: one initial model + one RL updated model (memory required for full parameters of two models)
            
            ![image.png](Week%203%20Notes/image%2019.png)
            
            - The weights of the reference model are frozen and are not updated.
            - KL divergence is a statistical measure of how different two probability distributions are. You can use it to compare the completions of the two models and determine how much the updated model has diverged from the reference. Keep in mind that this is still a relatively compute expensive process. You will almost always benefit from using GPUs.
        - Case2: combining PEFT adapter (one initial model can be used twice, only one set of parameters is required to be stored)
            
            ![image.png](Week%203%20Notes/image%2020.png)
            
            - This means that you can reuse the same underlying LLM for both the reference model and the PPO model, which you update with a trained PEFT parameters. This reduces the memory footprint during training by approximately half.
- Evaluate the human-aligned LLM
    
    ![image.png](Week%203%20Notes/image%2021.png)
    
- Beyond RLHF - scaling human feedback using AI
- In RLHF, the human effort required to produce the trained reward model in the first place is huge.
- One idea to overcome these limitations is to scale through model self supervision. Constitutional AI is one approach of scale supervision.
    - First proposed in 2022 by researchers at Anthropic, Constitutional AI is a method for training models using a set of rules and principles that govern the model's behavior. Together with a set of sample prompts, these form the constitution. You then train the model to self critique and revise its responses to comply with those principles.
    - Constitutional AI is useful not only for scaling feedback, it can also help address some unintended consequences of RLHF. For example, depending on how the prompt is structured, an aligned model may end up revealing harmful information as it tries to provide the most helpful response it can.
        
        ![image.png](Week%203%20Notes/image%2022.png)
        
        - Providing the model with a set of constitutional principles can help the model balance these competing interests and minimize the harm.
    - example rules from the research paper that Constitutional AI I asks LLMs to follow
        
        ![image.png](Week%203%20Notes/image%2023.png)
        
- When implementing the Constitutional AI method, you train your model in two distinct phases.
    
    ![image.png](Week%203%20Notes/image%2024.png)
    
    - In the first stage
        
        ![image.png](Week%203%20Notes/image%2025.png)
        
        - you carry out supervised learning, to start your prompt the model in ways that try to get it to generate harmful responses, this process is called red teaming.
        - You then ask the model to critique its own harmful responses according to the constitutional principles and revise them to comply with those rules.
        - Once done, you'll fine-tune the model using the pairs of red team prompts and the revised constitutional responses.
        
        ![image.png](Week%203%20Notes/image%2026.png)
        
        - The original red team prompt, and this final constitutional response can then be used as training data. You'll build up a data set of many examples like this to create a fine-tuned LLM that has learned how to generate constitutional responses.
    - In the second stage
        - This stage is similar to RLHF, except that instead of human feedback, we now use feedback generated by a model. This is sometimes referred to as reinforcement learning from AI feedback or RLAIF.
        - Here you use the fine-tuned model from the previous step to generate a set of responses to your prompt.
        - You then ask the model which of the responses is preferred according to the constitutional principles.
        - The result is a model generated preference dataset that you can use to train a reward model. With this reward model, you can now fine-tune your model further using a reinforcement learning algorithm like PPO, as discussed earlier.
## Lab3 Notes (https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/main/Week-3/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb)
- In this notebook, you will fine-tune a FLAN-T5 model to generate less toxic content with Meta AI's hate speech reward model. The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. You will use Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.
- Part1 set up dependencies
    
    ```jsx
    %pip install --upgrade pip
    %pip install --disable-pip-version-check \
        torch==1.13.1 \
        torchdata==0.5.1 --quiet
    
    %pip install \
        transformers==4.27.2 \
        datasets==2.11.0 \
        evaluate==0.4.0 \
        rouge_score==0.1.2 \
        peft==0.3.0 --quiet
    
    # Installing the Reinforcement Learning library directly from github.
    %pip install git+https://github.com/lvwerra/trl.git@25fa1bd    
    
    #---------------
    
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
    from datasets import load_dataset
    from peft import PeftModel, PeftConfig, LoraConfig, TaskType
    
    # trl: Transformer Reinforcement Learning library
    from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
    from trl import create_reference_model
    from trl.core import LengthSampler
    
    import torch
    import evaluate
    
    import numpy as np
    import pandas as pd
    
    # tqdm library makes the loops show a smart progress meter.
    from tqdm import tqdm
    tqdm.pandas()
    ```
    
    - new library introduced:
        - trl - give us access to PPO
        - AutoModelForSequenceClassification - This is what we're going to use to load our Facebook binary classifier, sequenceClassifier. Which when we give it a string of text or a sequence of text, it'll tell us whether or not that text contains hate speech or not, with a particular distribution across not hate or hate.
- Part2 Load dataset&model, prepare reward model and toxicity evaluator
    - build_dataset (splitting to train and test parts)
        
        ```jsx
        def build_dataset(model_name,
                            dataset_name,
                            input_min_text_length, 
                            input_max_text_length):
        
            """
            Preprocess the dataset and split it into train and test parts.
        
            Parameters:
            - model_name (str): Tokenizer model name.
            - dataset_name (str): Name of the dataset to load.
            - input_min_text_length (int): Minimum length of the dialogues.
            - input_max_text_length (int): Maximum length of the dialogues.
                
            Returns:
            - dataset_splits (datasets.dataset_dict.DatasetDict): Preprocessed dataset containing train and test parts.
            """
            
            # load dataset (only "train" part will be enough for this lab).
            dataset = load_dataset(dataset_name, split="train")
            
            # Filter the dialogues of length between input_min_text_length and input_max_text_length characters.
            dataset = dataset.filter(lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length, batched=False)
        
            # Prepare tokenizer. Setting device_map="auto" allows to switch between GPU and CPU automatically.
            tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
            
            def tokenize(sample):
                
                # Wrap each dialogue with the instruction.
                prompt = f"""
        Summarize the following conversation.
        
        {sample["dialogue"]}
        
        Summary:
        """
                sample["input_ids"] = tokenizer.encode(prompt)
                
                # This must be called "query", which is a requirement of our PPO library.
                sample["query"] = tokenizer.decode(sample["input_ids"])
                return sample
        
            # Tokenize each dialogue.
            dataset = dataset.map(tokenize, batched=False)
            dataset.set_format(type="torch")
            
            # Split the dataset into train and test parts.
            dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)
        
            return dataset_splits
        
        dataset = build_dataset(model_name=model_name,
                                dataset_name=huggingface_dataset_name,
                                input_min_text_length=200, 
                                input_max_text_length=1000)
        
        print(dataset)
        ```
        
    - Load the model from lab2 stored at aws s3
        
        ```python
        !aws s3 cp --recursive s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint-from-s3/ 
        ```
        
    - Adding PEFT adapter to have PEFT model
        - Add the adapter to the original FLAN-T5 model. In the previous lab you were adding the fully trained adapter only for inferences, so there was no need to pass LoRA configurations doing that. Now you need to pass them to the constructed PEFT model, also putting `is_trainable=True`.
        
        ```python
        !ls -alh ./peft-dialogue-summary-checkpoint-from-s3/adapter_model.bin
        
        #------
        
        lora_config = LoraConfig(
            r=32, # Rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                                        torch_dtype=torch.bfloat16)
        
        peft_model = PeftModel.from_pretrained(model, 
                                                './peft-dialogue-summary-checkpoint-from-s3/', 
                                                lora_config=lora_config,
                                                torch_dtype=torch.bfloat16, 
                                                device_map="auto",                                       
                                                is_trainable=True)
        
        ```
        
    - prepare the Proximal Policy Optimization (PPO) model passing the instruct-fine-tuned PEFT model to it.
        
        ```python
        ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
                                                                        torch_dtype=torch.bfloat16,
                                                                        is_trainable=True)
        ```
        
    - The original PPO model was copied as the reference model and later being used to calculate KL divergence value. The reference model will represent the LLM before detoxification. None of the parameters of the reference model will be updated during PPO training.
        
        ```python
        ref_model = create_reference_model(ppo_model)
        ```
        
    - Prepare Reward Model
        - In the [previous section](https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/5563fae447eb09be0cd1b8970fa8b07fa3b88042/Week-3/#2.1) the original policy is based on the instruct PEFT model - this is the LLM before detoxification. Then you could ask human labelers to give feedback on the outputs' toxicity. However, it can be expensive to use them for the entire fine-tuning process. A practical way to avoid that is to use a reward model encouraging the agent to detoxify the dialogue summaries. The intuitive approach would be to do some form of sentiment analysis across two classes (`nothate` and `hate`) and give a higher reward if there is higher a chance of getting class `nothate` as an output.
        - You will use [Meta AI's RoBERTa-based hate speech model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) for the reward model. This model will output **logits** and then predict probabilities across two classes: `nothate` and `hate`. The logits of the output `nothate` will be taken as a positive reward. Then, the model will be fine-tuned with PPO using those reward values.
        - Create the instance of the required model class for the RoBERTa model. You also need to load a tokenizer to test the model. Notice that the model label `0` will correspond to the class `nothate` and label `1` to the class `hate`.
            - when doing the PPO, it’s important to make sure which logits index is the one to optimize. In this case, logits index 0 is the one which represents for ‘not hate’.
            
            ```python
            toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
            toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name, device_map="auto")
            toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name, device_map="auto")
            
            #----------
            non_toxic_text = "#Person 1# tells Tommy that he didn't like the movie."
            
            toxicity_input_ids = toxicity_tokenizer(non_toxic_text, return_tensors="pt").input_ids
            
            logits = toxicity_model(input_ids=toxicity_input_ids).logits
            print(f'logits [not hate, hate]: {logits.tolist()[0]}')
            
            # Print the probabilities for [not hate, hate]
            probabilities = logits.softmax(dim=-1).tolist()[0]
            print(f'probabilities [not hate, hate]: {probabilities}')
            
            # get the logits for "not hate" - this is the reward!
            not_hate_index = 0
            nothate_reward = (logits[:, not_hate_index]).tolist()
            print(f'reward (high): {nothate_reward}')
            ```
            
            ![image.png](Week%203%20Notes/image%2027.png)
            
            ![image.png](Week%203%20Notes/image%2028.png)
            
        - HuggingFace Inference pipeline - Setup Hugging Face inference pipeline to simplify the code for the toxicity reward model:
            - The function of it is we don't have to call all those low level model, e.g. generate and do the tokenizer and all that stuff, this pipeline will actually do it all for us.
            
            ```python
            device = 0 if torch.cuda.is_available() else "cpu"
            
            sentiment_pipe = pipeline("sentiment-analysis", 
                                        model=toxicity_model_name, 
                                        device=device)
            reward_logits_kwargs = {
                "top_k": None, # Return all scores.
                "function_to_apply": "none", # Set to "none" to retrieve raw logits.
                "batch_size": 16
            }
            
            reward_probabilities_kwargs = {
                "top_k": None, # Return all scores.
                "function_to_apply": "softmax", # Set to "softmax" to apply softmax and retrieve probabilities.
                "batch_size": 16
            }
            
            print("Reward model output:")
            print("For non-toxic text")
            print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
            print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))
            print("For toxic text")
            print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
            print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))
            ```
            
            ![image.png](Week%203%20Notes/image%2029.png)
            
    - Evaluation toxicity
        - The **toxicity score** is a decimal value between 0 and 1 where 1 is the highest toxicity.
            
            ```python
            toxicity_evaluator = evaluate.load("toxicity", 
                                                toxicity_model_name,
                                                module_type="measurement",
                                                toxic_label="hate")
            ```
            
        - This evaluator can be used to compute the toxicity of the dialogues prepared in section [2.1](https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/5563fae447eb09be0cd1b8970fa8b07fa3b88042/Week-3/#2.1). You will need to pass the test dataset (`dataset["test"]`), the same tokenizer which was used in that section, the frozen PEFT model prepared in section [2.2](https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/5563fae447eb09be0cd1b8970fa8b07fa3b88042/Week-3/#2.2), and the toxicity evaluator. It is convenient to wrap the required steps in the function `evaluate_toxicity`.
            
            ```python
            def evaluate_toxicity(model, 
                                    toxicity_evaluator, 
                                    tokenizer, 
                                    dataset, 
                                    num_samples):
                
                """
                Preprocess the dataset and split it into train and test parts.
            
                Parameters:
                - model (trl model): Model to be evaluated.
                - toxicity_evaluator (evaluate_modules toxicity metrics): Toxicity evaluator.
                - tokenizer (transformers tokenizer): Tokenizer to be used.
                - dataset (dataset): Input dataset for the evaluation.
                - num_samples (int): Maximum number of samples for the evaluation.
                    
                Returns:
                tuple: A tuple containing two numpy.float64 values:
                - mean (numpy.float64): Mean of the samples toxicity.
                - std (numpy.float64): Standard deviation of the samples toxicity.
                """
            
                max_new_tokens=100
            
                toxicities = []
                input_texts = []
                for i, sample in tqdm(enumerate(dataset)):
                    input_text = sample["query"]
            
                    if i > num_samples:
                        break
                        
                    input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
                    
                    generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                                            tok_k=0.0,
                                                            top_p=1.0,
                                                            do_sample=True)
            
                    response_token_ids = model.generate(input_ids=input_ids,
                                                        generation_config=generation_config)
                    
                    generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
                    
                    toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + generated_text)])
            
                    toxicities.extend(toxicity_score["toxicity"])
            
                # Compute mean & std using np.
                mean = np.mean(toxicities)
                std = np.std(toxicities)
                    
                return mean, std
            ```
            
        - And now perform the calculation of the model toxicity before fine-tuning/detoxification:
            
            ```python
            tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
            
            mean_before_detoxification, std_before_detoxification = evaluate_toxicity(model=ref_model, 
                                                                                        toxicity_evaluator=toxicity_evaluator, 
                                                                                        tokenizer=tokenizer, 
                                                                                        dataset=dataset["test"], 
                                                                                        num_samples=10)
            
            print(f'toxicity [mean, std] before detox: [{mean_before_detoxification}, {std_before_detoxification}]')
            ```
            
            ![image.png](Week%203%20Notes/image%2030.png)
            
- Part3 Fine-tuning to Detoxify (Optimize a RL policy against the reward model using Proximal Policy Optimization (PPO).)
    - Initialize PPOTrainer
        - collator function transform dict format
            
            ![image.png](Week%203%20Notes/image%2031.png)
            
        - Set up the configuration parameters. Load the `ppo_model` and the tokenizer. You will also load a frozen version of the model `ref_model`. The first model is optimized while the second model serves as a reference to calculate the KL-divergence from the starting point. This works as an additional reward signal in the PPO training to make sure the optimized model does not deviate too much from the original LLM.
        
        ```python
        learning_rate=1.41e-5
        max_ppo_epochs=1
        mini_batch_size=4
        batch_size=16
        
        config = PPOConfig(
            model_name=model_name,    
            learning_rate=learning_rate,
            ppo_epochs=max_ppo_epochs,
            mini_batch_size=mini_batch_size,
            batch_size=batch_size
        )
        
        ppo_trainer = PPOTrainer(config=config, 
                                    model=ppo_model, 
                                    ref_model=ref_model, 
                                    tokenizer=tokenizer, 
                                    dataset=dataset["train"], 
                                    data_collator=collator)
        ```
        
    - Fine-tune the model
        
        The fine-tuning loop consists of the following main steps:
        
        1. Get the query responses from the policy LLM (PEFT model).
        2. Get sentiments for query/responses from hate speech RoBERTa model.
        3. Optimize policy with PPO using the (query, response, reward) triplet.
        
        The operation is running if you see the following metrics appearing:
        
        - `objective/kl`: minimize kl divergence,
        - `ppo/returns/mean`: maximize mean returns,
        - `ppo/policy/advantages_mean`: maximize advantages.
        
        ```python
        output_min_length = 100
        output_max_length = 400
        output_length_sampler = LengthSampler(output_min_length, output_max_length)
        
        generation_kwargs = {
            "min_length": 5,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True
        }
        
        reward_kwargs = {
            "top_k": None, # Return all scores.
            "function_to_apply": "none", # You want the raw logits without softmax.
            "batch_size": 16
        }
        
        max_ppo_steps = 10
        
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            # Break when you reach max_steps.
            if step >= max_ppo_steps:
                break   
        
            prompt_tensors = batch["input_ids"]
        
            # Get response from FLAN-T5/PEFT LLM.
            summary_tensors = []
        
            for prompt_tensor in prompt_tensors:
                max_new_tokens = output_length_sampler()        
                    
                generation_kwargs["max_new_tokens"] = max_new_tokens
                summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
                
                summary_tensors.append(summary.squeeze()[-max_new_tokens:])
                
            # This needs to be called "response".
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]
        
            # Compute reward outputs.
            query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]    
            rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)
        
            # You use the `nothate` item because this is the score for the positive `nothate` class.
            reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]    
        
            # Run PPO step.
            stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
            ppo_trainer.log_stats(stats, batch, reward_tensors)
            
            print(f'objective/kl: {stats["objective/kl"]}')
            print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
            print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
            print('-'.join('' for x in range(100)))
        ```
        
    - Evaluate the before and after model quantitatively and qualitatively
        
        ```python
        mean_after_detoxification, std_after_detoxification = evaluate_toxicity(model=ppo_model, 
                                                                                toxicity_evaluator=toxicity_evaluator, 
                                                                                tokenizer=tokenizer, 
                                                                                dataset=dataset["test"], 
                                                                                num_samples=10)
        print(f'toxicity [mean, std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')
        ```
        
        ![image.png](Week%203%20Notes/image%2032.png)
        


---

## For the Generative AI project lifecycle, we are now looking at the Application integration stage

![image.png](Week%203%20Notes/image%2033.png)

- A number of important questions to ask about this stage:
    - The first set is related to how your LLM will function in deployment.
        - how fast do you need your model to generate completions?
        - What compute budget do you have available?
        - And are you willing to trade off model performance for improved inference speed or lower storage?
    - The second set of questions is tied to additional resources that your model may need.
        - Do you intend for your model to interact with external data or other applications?
        - And if so, how will you connect to those resources?
        - Lastly, there's the question of how your model will be consumed. What will the intended application or API interface that your model will be consumed through look like?
## Model optimizations for deployment
- Large language models present inference challenges in terms of computing and storage requirements, as well as ensuring low latency for consuming applications. These challenges persist whether you're deploying on premises or to the cloud, and become even more of an issue when deploying to edge devices.
- One of the primary ways to improve application performance is to reduce the size of the LLM. This can allow for quicker loading of the model, which reduces inference latency. However, the challenge is to reduce the size of the model while still maintaining model performance.
- Three techniques covered:
    
    ![image.png](Week%203%20Notes/image%2034.png)
    
    - Distillation uses a larger model, the teacher model, to train a smaller model, the student model. You then use the smaller model for inference to lower your storage and compute budget.
    - Similar to quantization aware training, post training quantization transforms a model's weights to a lower precision representation
    - Model Pruning, removes redundant model parameters that contribute little to the model's performance.
- Distillation
    - The student model learns to statistically mimic the behavior of the teacher model, either just in the final prediction layer or in the model's hidden layers as well.
    
    ![image.png](Week%203%20Notes/image%2035.png)
    
    - Process
        - You start with your fine tune LLM as your teacher model and create a smaller LLM for your student model. You freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model.
        - The knowledge distillation between teacher and student model is achieved by minimizing a loss function called the distillation loss. To calculate this loss, distillation uses the probability distribution over tokens that is produced by the teacher model's softmax layer.
        - Now, the teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function.
            - GPT notes on why we don’t want the teacher distribution to be closely matching the ground truth?
                
                This goes deep into the heart of why **knowledge distillation** works.
                
                Let’s break it down:
                
                ---
                
                ## 🌡️ What’s the deal with temperature in distillation?
                
                The **temperature** in softmax is used to **soften** the probability distribution.
                
                This helps the student **learn not just the correct answer**, but also the teacher’s **beliefs about all other answers**.
                
                ---
                
                ## 🧠 First, why not just use the teacher’s raw output?
                
                Let’s look at a simple example.
                
                Imagine your **teacher model** sees this input and produces:
                
                ```
                Token probabilities (with T=1, regular softmax):
                "cat": 0.98
                "dog": 0.01
                "car": 0.005
                "cactus": 0.005
                
                ```
                
                Pretty spiky. The teacher is *very* confident it's "cat".
                
                Now the **student** sees only:
                
                > "Oh, ‘cat’ is the answer. Got it. I’ll just focus on predicting that."
                > 
                
                This teaches the student **what to think**, but not **how confident** or **why**.
                
                ---
                
                ## 🔥 What if we raise the temperature?
                
                Say we use a **higher temperature** (e.g. T = 2.0). The softmax becomes:
                
                ```
                "cat": 0.60
                "dog": 0.20
                "car": 0.10
                "cactus": 0.10
                
                ```
                
                Now the distribution is **softer**. It says:
                
                > "Sure, ‘cat’ is the best, but ‘dog’, ‘car’, and ‘cactus’ are still kinda okay guesses."
                > 
                
                This gives the student **richer signal**. It learns not just the final answer, but the **relative similarities** between options.
                
                ---
                
                ## 🧪 Why is this good?
                
                Because the **teacher knows more** than just the label. It encodes:
                
                - semantic similarity,
                - uncertainty,
                - relationships between words.
                
                If you only train the student on hard labels or sharp distributions, it learns to **copy outputs** instead of understanding the **latent knowledge**.
                
                The softened distributions help the student:
                
                - generalize better,
                - converge faster,
                - and avoid overfitting to only "hard truths".
                
                ---
                
                ## 🤔 But why do teachers produce sharp distributions?
                
                Great observation. The teacher has been fine-tuned and is **very confident**, so it gives **very spiky predictions** (close to 1-hot vectors).
                
                That’s actually a **problem for learning**: the gradients from such predictions are small or noisy.
                
                Using a higher **temperature smooths this out** and makes the training signal more informative.
                
                ---
                
                ## ✅ Summary:
                
                - **Without temperature (T=1)**: the teacher output is too sharp → student only learns "pick the most likely token".
                - **With temperature (T>1)**: the teacher output is softened → student learns **why** the teacher favors certain tokens, not just which one.
                
                Distillation isn’t just copying — it’s teaching how to think like the teacher.
                
                ---
                
                Let me know if you want to walk through this with code, or visualize how the softmax changes with temperature!
                
        - With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked.
        - Distillation refers to the student model outputs as the hard predictions and hard labels.
        - The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via back propagation.
    - The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model.
    - In practice, distillation is not as effective for generative decoder models. It's typically more effective for encoder only models, such as Bert that have a lot of representation redundancy.
- Quantization (specifically Post-Training Quantization (PTO))
    - This is diff from the Quantization Aware Training (QAT) introduced in week1. After a model is trained, you can perform post training quantization.
    
    ![image.png](Week%203%20Notes/image%2036.png)
    
    - PTQ transforms a model's weights to a lower precision representation, such as 16-bit floating point or 8-bit integer. To reduce the model size and memory footprint, as well as the compute resources needed for model serving, quantization can be applied to just the model weights or to both weights and activation layers.
    - In general, quantization approaches that include the activations can have a higher impact on model performance.
    - Quantization also requires an extra calibration step to statistically capture the dynamic range of the original parameter values.
    - Tradeoffs - sometimes quantization results in a small percentage reduction in model evaluation metrics. However, that reduction can often be worth the cost savings and performance gains.
- Pruning - reduce model size for inference by eliminating weights that are not contributing much to overall model performance.
    
    ![image.png](Week%203%20Notes/image%2037.png)
    
    - weights with values very close to or equal to zero.
    - Note that some pruning methods require full retraining of the model, while others fall into the category of parameter efficient fine tuning, such as LoRA. There are also methods that focus on post-training Pruning.
    - In practice, there may not be much impact on the size and performance if only a small percentage of the model weights are close to zero.
## Generative AI Project Lifecycle Cheat Sheet
- this cheat sheet provide some indication of the time and effort required for each phase of work.

![image.png](Week%203%20Notes/image%2038.png)

- pre-training a large language model can be a huge effort. This stage is the most complex you'll face because of the model architecture decisions, the large amount of training data required, and the expertise needed.  However, in general, you will start your development work with an existing foundation model. You'll probably be able to skip this stage.
- If you're working with a foundation model, you'll likely start to assess the model's performance through prompt engineering, which requires less technical expertise, and no additional training of the model.
- If your model isn't performing as you need, you'll next think about prompt tuning and fine tuning. Depending on your use case, performance goals, and compute budget, the methods you'll try could range from full fine-tuning to parameter efficient fine tuning techniques like LoRA or prompt tuning. Some level of technical expertise is required for this work. But since fine-tuning can be very successful with a relatively small training dataset, this phase could potentially be completed in a single day.
- Aligning your model using reinforcement learning from human feedback can be done quickly, once you have your train reward model. However, if you have to train a reward model from scratch, it could take a long time because of the effort involved to gather human feedback.

---

## Other challenges with LLM outside training part

![image.png](Week%203%20Notes/image%2039.png)

- The internal knowledge held by a model cuts off at the moment of pre-training.
- Models can also struggle with complex math. LLMs do not carry out mathematical operations. They are still just trying to predict the next best token based on their training, and as a result, can easily get the answer wrong.
- hallucination - LLM generates seems legit answers but are actually made up and wrong.
## Connect to external data sources and applications to solve above challenges

![image.png](Week%203%20Notes/image%2040.png)

- One implementation example is Langchain (explained later)
## How to connect LLM to external data sources - Retrieval Augmented Generation (RAG)
- A framework for building LLM powered systems that make use of external data sources and applications to overcome some of the limitations of these models.
- RAG can overcome knowledge cut-off issue in a much cheaper way comparing to constant re-training.
    - RAG is useful in any case where you want the language model to have access to data that it may not have seen. This could be new information documents not included in the original training data, or proprietary knowledge stored in your organization's private databases.
    - Providing your model with external information, can improve both the relevance and accuracy of its completions.
- RAG isn't a specific set of technologies, but rather a framework for providing LLMs access to data they did not see during training. A number of different implementations exist, and the one you choose will depend on the details of your task and the format of the data you have to work with.
    - Example walk-through (implementation from Facebook 2020, one of the earliest)
        
        ![image.png](Week%203%20Notes/image%2041.png)
        
        - At the heart of this implementation is a model component called the Retriever, which consists of a query encoder and an external data source. The encoder takes the user's input prompt and encodes it into a form that can be used to query the data source.
        - These two components are trained together to find documents within the external data that are most relevant to the input query. The Retriever returns the best single or group of documents from the data source and combines the new information with the original user query.
        - The new expanded prompt is then passed to the language model, which generates a completion that makes use of the data.
- RAG also helps with overcoming hallucinating.
- RAG architectures can be used to integrate multiple types of external information sources.
    
    ![image.png](Week%203%20Notes/image%2042.png)
    
    - By encoding the user input prompt as a SQL query, RAG can also interact with databases.
    - Vector Store - which contains vector representations of text.
        - This is a particularly useful data format for language models, since internally they work with vector representations of language to generate text.
        - Vector stores enable a fast and efficient kind of relevant search based on similarity.
- Other key considerations for implementing RAG
    - Size of context window
        
        ![image.png](Week%203%20Notes/image%2043.png)
        
        - Most text sources are too long to fit into the limited context window of the model, which is still at most just a few thousand tokens
        - Instead, the external data sources are chopped up into many chunks, each of which will fit in the context window. Packages like Langchain can handle this work for you.
    - data format to allow relevance assessed at inference time (via embedding vectors)
        
        ![image.png](Week%203%20Notes/image%2044.png)
        
        ![image.png](Week%203%20Notes/image%2045.png)
        
        - RAG methods take the small chunks of external data and process them through the large language model, to create embedding vectors for each. These new representations of the data can be stored in structures called vector stores, which allow for fast searching of datasets and efficient identification of semantically related text.
            - Vector databases - are a particular implementation of a vector store where each vector is also identified by a key. This can allow, for instance, the text generated by RAG to also include a citation for the document from which it was received.
                
                ![image.png](Week%203%20Notes/image%2046.png)
                
## How LLM interact with external applications
- In general, connecting LLMs to external applications allows the model to interact with the broader world, extending their utility beyond language tasks.
    
    ![image.png](Week%203%20Notes/image%2047.png)
    
    - LLMs can be used to trigger actions when given the ability to interact with APIs.
    - LLMs can also connect to other programming resources. e.g. Python interpreter.
- It's important to note that prompts and completions are at the very heart of these workflows. The actions that the app will take in response to user requests will be determined by the LLM, which serves as the application's reasoning engine.
    
    ![image.png](Week%203%20Notes/image%2048.png)
    
    - First, the model needs to be able to generate a set of instructions so that the application knows what actions to take. These instructions need to be understandable and correspond to allowed actions.
    - Second, the completion needs to be formatted in a way that the broader application can understand.
    - Lastly, the model may need to collect information that allows it to validate an action.
    - Structuring the prompts in the correct way is important for all of these tasks and can make a huge difference in the quality of a plan generated or the adherence to a desired output format specification
## Chain-of-thought (help with complex reasoning)
- As you saw, it is important that LLMs can reason through the steps that an application must take, to satisfy a user request.  Unfortunately, complex reasoning can be challenging for LLMs. And chain-of-thought is the way to help, by having the model solve problems more like human.
    
    ![image.png](Week%203%20Notes/image%2049.png)
    
    - It works by including a series of intermediate reasoning steps into any examples that you use for one or few-shot inference. By structuring the examples in this way, you're essentially teaching the model how to reason through the task to reach a solution.
    - You can use chain of thought prompting to help LLMs improve their reasoning of other types of problems too, in addition to arithmetic.
- While this can greatly improve the performance of your model, the limited math skills of LLMs can still cause problems if your task requires accurate calculations, like totaling sales on an e-commerce site, calculating tax, or applying a discount.  So coming next, is having LLM talk to a program to help with math ability.
## Program-aided language models (PAL)
- This work first presented by Luyu Gao and collaborators at Carnegie Mellon University in 2022, pairs an LLM with an external code interpreter to carry out calculations.
    - The method makes use of chain of thought prompting to generate executable Python scripts. The scripts that the model generates are passed to an interpreter to execute.
    - The strategy behind PAL is to have the LLM generate completions where reasoning steps are accompanied by computer code. This code is then passed to an interpreter to carry out the calculations necessary to solve the problem.
    - You specify the output format for the model by including examples for one or few short inference in the prompt.
    - example
        
        ![image.png](Week%203%20Notes/image%2050.png)
        
        - the chain of thought reasoning steps are shown in blue and the Python code is shown in pink.
- How the PAL framework enables an LLM to interact with an external interpreter
    
    ![image.png](Week%203%20Notes/image%2051.png)
    
    - To prepare for inference with PAL, you'll format your prompt to contain one or more examples. Each example should contain a question followed by reasoning steps in lines of Python code that solve the problem.
    - Next, you will append the new question that you'd like to answer to the prompt template. Your resulting PAL formatted prompt now contains both the example and the problem to solve.
    - Next, you'll pass this combined prompt to your LLM, which then generates a completion that is in the form of a Python script having learned how to format the output based on the example in the prompt.
    - You can now hand off the script to a Python interpreter, which you'll use to run the code and generate an answer.
    
    ![image.png](Week%203%20Notes/image%2052.png)
    
    - You'll now append the text containing the answer, which you know is accurate because the calculation was carried out in Python to the PAL formatted prompt you started with. By this point you have a prompt that includes the correct answer in context.
    
    ![image.png](Week%203%20Notes/image%2053.png)
    
    - Now when you pass the updated prompt to the LLM, it generates a completion that contains the correct answer.
- Orchestrator - Automate above process to avoid passing info back and forth between the LLM & interpreter by hand
    
    ![image.png](Week%203%20Notes/image%2054.png)
    
    - The orchestrator shown here as the yellow box is a technical component that can manage the flow of information and the initiation of calls to external data sources or applications. It can also decide what actions to take based on the information contained in the output of the LLM.
    - The LLM is your application's reasoning engine. Ultimately, it creates the plan that the orchestrator will interpret and execute.
## ReAct: Combining reasoning and action
- Used for more complex workflows, perhaps in including interactions with multiple external data sources and applications.
    - ReAct is a prompting strategy that combines chain of thought reasoning with action planning. The framework was proposed by researchers at Princeton and Google in 2022.
        
        ![image.png](Week%203%20Notes/image%2055.png)
        
- ReAct uses structured examples to show a large language model how to reason through a problem and decide on actions to take that move it closer to a solution.
    - The example prompts start with a question that will require multiple steps to answer. Then includes a related thought, action, observation trio of string
        
        ![image.png](Week%203%20Notes/image%2056.png)
        
        ![image.png](Week%203%20Notes/image%2057.png)
        
        ![image.png](Week%203%20Notes/image%2058.png)
        
        - The action is formatted using the specific square bracket notation you see here, so that the model will format its completions in the same way. The Python interpreter searches for this code to trigger specific API actions.
        
        ![image.png](Week%203%20Notes/image%2059.png)
        
        - this is where the new information provided by the external search is brought into the context of the prompt for the model to interpret.
        - The prompt then repeats the cycle as many times as is necessary to obtain the final answer.
        
        ![image.png](Week%203%20Notes/image%2060.png)
        
        ![image.png](Week%203%20Notes/image%2061.png)
        
- It's important to note that in the ReAct framework, the LLM can only choose from a limited number of actions that are defined by a set of instructions that is pre-pended to the example prompt text.
- Full text of instructions:
    
    ![image.png](Week%203%20Notes/image%2062.png)
    
    - First, the task is defined, telling the model to answer a question using the prompt structure you just explored in detail.
    - Next, the instructions give more detail about what is meant by thought and then specifies that the action step can only be one of three types.
    - It is critical to define a set of allowed actions when using LLMs to plan tasks that will power applications. LLMs are very creative, and they may propose taking steps that don't actually correspond to something that the application can do.
- Building ReAct prompt - Put all pieces together for inference
    
    ![image.png](Week%203%20Notes/image%2063.png)
    
    - You'll start with the ReAct example prompt. Note that depending on the LLM you're working with, you may find that you need to include more than one example and carry out future inference.
    - Next, you'll pre-pend the instructions at the beginning of the example and then insert the question you want to answer at the end.
    - The full prompt now includes all of these individual pieces, and it can be passed to the LLM for inference.
- LangChain framework - provides you with modular pieces that contain the components necessary to work with LLMs.
    
    ![image.png](Week%203%20Notes/image%2064.png)
    
    - Components include
        - prompt templates for many different use cases that you can use to format both input examples and model completions.
        - memory that you can use to store interactions with an LLM.
        - pre-built tools that enable you to carry out a wide variety of tasks, including calls to external datasets and various APIs.
        - Connecting a selection of these individual components together results in a chain. The creators of LangChain have developed a set of predefined chains that have been optimized for different use cases, and you can use these off the shelf to quickly get your app up and running.
        - In case you can’t use a pre-determined chain, agent component can be used to interpret the input from the user and determine which tool or tools to use to complete the task.
        - LangChain currently includes agents for both PAL and ReAct, among others.
- The ability of the model to reason well and plan actions depends on its scale.
    - Larger models are generally your best choice for techniques that use advanced prompting, like PAL or ReAct.
    - Smaller models may struggle to understand the tasks in highly structured prompts and may require you to perform additional fine tuning to improve their ability to reason and plan. This could slow down your development process.
        - Instead, if you start with a large, capable model and collect lots of user data in deployment, you may be able to use it to train and fine tune a smaller model that you can switch to at a later time.

---

## LLM application architectures

![image.png](Week%203%20Notes/image%2065.png)

- Infrastructure layer
    - This layer provides the compute, storage, and network to serve up your LLMs, as well as to host your application components. You can make use of your on-premises infrastructure for this or have it provided for you via on-demand and pay-as-you-go Cloud services.
- LLM models + external sources + stored outputs
    - These could include foundation models, as well as the models you have adapted to your specific task. The models are deployed on the appropriate infrastructure for your inference needs.
    - You may also have the need to retrieve information from external sources, such as those discussed in the retrieval augmented generation section.
    - Your application will return the completions from your large language model to the user or consuming application. Depending on your use case, you may need to implement a mechanism to capture and store the outputs.
        - For example, you could build the capacity to store user completions during a session to augment the fixed contexts window size of your LLM.
        - You can also gather feedback from users that may be useful for additional fine-tuning, alignment, or evaluation as your application matures.
- LLM Tools & Frameworks
    - As an example, you can use LangChain’s built-in libraries to implement techniques like PAL, ReAct or chain-of-thought prompting.
    - You may also utilize model hubs which allow you to centrally manage and share models for use in applications.
- Application Interfaces
    - Some type of user interface that the application will be consumed through, such as a website or a rest API.
    - This layer is where you'll also include the security components required for interacting with your application.

---

## Week3 Resources: https://www.coursera.org/learn/generative-ai-with-llms/supplement/89tR9/week-3-resources

---

## responsible AI

![image.png](Week%203%20Notes/image%2066.png)

![image.png](Week%203%20Notes/image%2067.png)

![image.png](Week%203%20Notes/image%2068.png)

![image.png](Week%203%20Notes/image%2069.png)

![image.png](Week%203%20Notes/image%2070.png)

- ongoing research field related to responsible AI:
    - There's water marking and fingerprinting which are ways to include almost like a stamp or a signature in a piece of content or data so that we can always trace back.
    - creating models that help determine if content was created with gentle AI is also a budding field of research.
## other ongoing research areas

![image.png](Week%203%20Notes/image%2071.png)