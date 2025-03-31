# Week 2 Notes

- [Week 2 Notes](#week-2-notes)
  - [Key questions on this week](#key-questions-on-this-week)
  - [Instruction fine-tuning](#instruction-fine-tuning)
  - [Fine-tuning on a single task](#fine-tuning-on-a-single-task)
  - [Multi-task instruction fine-tuning](#multi-task-instruction-fine-tuning)
  - [Model Evaluation - ROUGE \& BLEU](#model-evaluation---rouge--bleu)
  - [Benchmarks](#benchmarks)
  - [PEFT Intro](#peft-intro)
  - [PEFT techniques 1: LoRA (Low-rank Adaptation)](#peft-techniques-1-lora-low-rank-adaptation)
  - [PEFT techniques 2: Soft prompts (method: prompt tuning)](#peft-techniques-2-soft-prompts-method-prompt-tuning)
  - [Summary of the two PEFT techs: Both methods enable you to fine tune models with the potential for improved performance on your tasks while using much less compute than full fine tuning methods. LoRA is broadly used in practice because of the comparable performance to full fine tuning for many tasks and data sets.](#summary-of-the-two-peft-techs-both-methods-enable-you-to-fine-tune-models-with-the-potential-for-improved-performance-on-your-tasks-while-using-much-less-compute-than-full-fine-tuning-methods-lora-is-broadly-used-in-practice-because-of-the-comparable-performance-to-full-fine-tuning-for-many-tasks-and-data-sets)
  - [Lab2 notes (https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/main/Week-2/Lab\_2\_fine\_tune\_generative\_ai\_model.ipynb)](#lab2-notes-httpsgithubcomryota-kawamuragenerative-ai-with-llmsblobmainweek-2lab_2_fine_tune_generative_ai_modelipynb)
  - [Week2 Resources](#week2-resources)


## Key questions on this week
    - What is and how to do fine-tuning with instructions?
    - How to deal with the forgetting issue regarding the pre-training part of memory?
    - How to do fine-tuning for specialized app? (Parameter-efficient-fine-tuning) And what’s the benefit using it?
    - What's LoRA?
    - What are the factors to consider for developers’ choice between large models v.s. smaller fine-tuned models?
## Instruction fine-tuning
    - Limitations of in-context learning
        - may not work for smaller models, even with 5 or 6 examples
        - the examples provided in one-shot/multi-shot take up valuable space in the context window, reducing the amount of room you have to include other useful information.
    - fine-tuning is supervised learning v.s. pre-training as the self-supervised learning
        - compare to pre-training, fine-tuning usually requires smaller datasets
        - data examples are prompt-completion pairs
        
        ![image.png](Week%202%20Notes/image.png)
        
    - One tech: Instruction fine tuning  - particularly good at improving model performance, and most common
        
        These prompt completion examples allow the model to learn to generate responses that follow the given instructions.
        
        ![image.png](Week%202%20Notes/image%201.png)
        
    - How it works?
        - Include a specific, fixed instruction part in the prompt followed by text (as X), and then provide it with corresponding completion (as Y).
        - It’s known as full fine-tuning which results in updates to all model parameters. Thus also requires the memory to store all related parameters.
            
            It is important to note that just like pre-training, full fine tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components that are being updated during training. So you can benefit from the memory optimization and parallel computing strategies that you learned about last week.
            
        - 1st step - prepare the training data
            
            There are many publicly available datasets that have been used to train earlier generations of language models. And developers have assembled prompt template libraries that can be used to take existing datasets, and turn them into instruction prompt datasets for fine-tuning.
            
            Prompt template libraries include many templates for different tasks and different data sets.  Examples with Amazon reviews dataset:
            
            ![image.png](Week%202%20Notes/image%202.png)
            
        - 2nd step - split into training/validation/test datasets as in ML
            
            ![image.png](Week%202%20Notes/image%203.png)
            
            ![image.png](Week%202%20Notes/image%204.png)
            
            - During fine tuning, you select prompts from your training data set and pass them to the LLM, which then generates completions.
            - Next, you compare the LLM completion with the response specified in the training data.
            - The output of an LLM is a probability distribution across tokens. So you can compare the distribution of the completion and that of the training label and use the standard cross-entropy function to calculate loss between the two token distributions. And then use the calculated loss to update your model weights in standard back-propagation. You'll do this for many batches of prompt completion pairs and over several epochs, update the weights so that the model's performance on the task improves.
            - Then can measure validation accuracy and test accuracy with corresponding dataset.
## Fine-tuning on a single task
    - For single task, only 500-1000 examples are needed to fine-tune to a good enough performance.
    - Catastrophic forgetting - One major disadvantage for fine-tuning on a single task
        
        Catastrophic forgetting happens because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on the single fine-tuning task, it can degrade performance on other tasks. 
        
        For example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks.
        
    - Ways to avoid Catastrophic forgetting
        
        ![image.png](Week%202%20Notes/image%205.png)
        
        - First of all, it's important to decide whether catastrophic forgetting actually impacts your use case. If all you need is reliable performance on the single task you fine-tuned on, it may not be an issue that the model can't generalize to other tasks.
        - Perform fine-tuning on multiple tasks at one time. Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train. (details later)
        - Parameter efficient fine-tuning(PEFT)instead of full fine-tuning.
            
            PEFT is a set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters. PEFT shows greater robustness to catastrophic forgetting since most of the pre-trained weights are left unchanged.
            
## Multi-task instruction fine-tuning
    - Train the model on this mixed dataset so that it can improve the performance of the model on all the tasks simultaneously, thus avoiding the issue of catastrophic forgetting.
        
        Over many epochs of training, the calculated losses across examples are used to update the weights of the model, resulting in an instruction tuned model that is learned how to be good at many different tasks simultaneously.
        
        ![image.png](Week%202%20Notes/image%206.png)
        
    - One drawback - it requires a lot of data. You may need as many as 50k-100k examples in your training set. But it can be worthwhile given the use case and performance.
    - One use case - FLAN family of models (fine-tuned language net)
        
        ![image.png](Week%202%20Notes/image%207.png)
        
        FLAN-T5, the FLAN instruction version of the T5 foundation model while FLAN-PALM is the FLAN instruction version of the palm foundation model.
        
        ![image.png](Week%202%20Notes/image%208.png)
        
        FLAN-T5 is a great general purpose instruct model. In total, it's been fine tuned on 473 datasets across 146 task categories. Those datasets are chosen from other models and papers as shown here. 
        
        - One prompt template example for Summarization tasks used in FLAN-T5 fine-tuning
            
            ![image.png](Week%202%20Notes/image%209.png)
            
            The template is actually comprised of several different instructions that all basically ask the model to do this same thing. Summarize a dialogue. Including different ways of saying the same instruction helps the model generalize and perform better.
            
        - If we want to improve FLAN-T5 on client-support type of conversation summary we can fine-tune the model with an additional round using those type of conversation data.
            
            ![image.png](Week%202%20Notes/image%2010.png)
            
            ![image.png](Week%202%20Notes/image%2011.png)
            
    - In practice, you'll get the most out of fine-tuning by using your company's own internal data.
    - One thing you need to think about when fine-tuning is how to evaluate the quality of your models completions - coming as the next part
## Model Evaluation - ROUGE & BLEU
    - evaluation is more challenging for LLM comparing to traditional ML model bcs the output is non-deterministic and we need language-based evaluation.
    - Two widely used evaluation metrics
        
        ![image.png](Week%202%20Notes/image%2012.png)
        
        - Some terms
            
            ![image.png](Week%202%20Notes/image%2013.png)
            
    - ROUGE (Recall-oriented under study for gisting evaluation)
        - compare to human-generated reference summaries
        - ROUGE-1 metric (based on unigram)
            - metrics similar to ML tasks using recall, precision, and F1.
            - It doesn’t count in the ordering of the words, can be deceptive.
            
            ![image.png](Week%202%20Notes/image%2014.png)
            
        - ROUGE-2 metric (based on bigrams, ordering is considered in a simple way)
            
            ![image.png](Week%202%20Notes/image%2015.png)
            
            With longer sentences, they're a greater chance that bigrams don't match, and the scores may be even lower. 
            
        - ROUGE-L (based on longest common subsequence)
            
            You can now use the LCS value to calculate the recall precision and F1 score, where the numerator in both the recall and precision calculations is the length of the longest common subsequence.
            
            ![image.png](Week%202%20Notes/image%2016.png)
            
        - As with all of the rouge scores, you need to take the values in context. You can only use the scores to compare the capabilities of models if the scores were determined for the same task.
            
            Rouge scores for different tasks are not comparable to one another. 
            
        - Cons for ROUGE way: it's possible for a bad completion to result in a good score.
            - One way you can counter this issue is by using a clipping function to limit the number of unigram matches to the maximum count for that unigram within the reference. In this case, there is one appearance of cold and the reference and so a modified precision with a clip on the unigram matches results in a dramatically reduced score.
            - However, you'll still be challenged if their generated words are all present, but just in a different order.
            
            ![image.png](Week%202%20Notes/image%2017.png)
            
        - Thus, how useful ROUGE score is will be dependent on the sentence, the sentence size and your use case.
        - Note that many language model libraries, for example, Hugging Face, include implementations of rouge score that you can use to easily evaluate the output of your model.
    - BLEU (bilingual evaluation understudy)
        - The score itself is calculated using the average precision over multiple n-gram sizes. Just like the Rouge-1 score that we looked at before, but calculated for a range of n-gram sizes and then averaged.
        - Calculating the BLEU score is easy with pre-written libraries from providers like Hugging Face.
            
            ![image.png](Week%202%20Notes/image%2018.png)
            
            As we get closer and closer to the original sentence, we get a score that is closer and closer to one. 
            
    - Both ROUGE and BLEU are useful for diagnostic evaluation
        - Both rouge and BLEU are quite simple metrics and are relatively low-cost to calculate. You can use them for simple reference as you iterate over your models, but you shouldn't use them alone to report the final evaluation of a large language model.
        - For overall evaluation of your model's performance, however, you will need to look at one of the evaluation benchmarks that have been developed by researchers.
## Benchmarks
    - In order to measure and compare LLMs more holistically, you can make use of pre-existing datasets, and associated benchmarks that have been established by LLM researchers specifically for this purpose.
    - Selecting the right evaluation dataset is vital, so that you can accurately assess an LLM's performance, and understand its true capabilities.
    - An important issue that you should consider is whether the model has seen your evaluation data during training.
    - Common benchmarks
        
        ![image.png](Week%202%20Notes/image%2019.png)
        
        They do this by designing or collecting datasets that test specific aspects of an LLM. 
        
        - GLUE(General Language Understanding Evaluation, in 2018)
            
            GLUE is a collection of natural language tasks, such as sentiment analysis and question-answering. GLUE was created to encourage the development of models that can generalize across multiple tasks.
            
            ![image.png](Week%202%20Notes/image%2020.png)
            
        - SuperGLUE (in 2019)
            
            ![image.png](Week%202%20Notes/image%2021.png)
            
            It consists of a series of tasks, some of which are not included in GLUE, and some of which are more challenging versions of the same tasks. SuperGLUE includes tasks such as multi-sentence reasoning, and reading comprehension. 
            
            Both the GLUE and SuperGLUE benchmarks have leaderboards that can be used to compare and contrast evaluated models. 
            
        - There is essentially an arms race between the emergent properties of LLMs, and the benchmarks that aim to measure them. Below are some more recent benchmarks.
        - MMLU & BIG-bench
            
            ![image.png](Week%202%20Notes/image%2022.png)
            
            - Massive Multitask Language Understanding, or MMLU, is designed specifically for modern LLMs. To perform well models must possess extensive world knowledge and problem-solving ability. Models are tested on elementary mathematics, US history, computer science, law, and more. In other words, tasks that extend way beyond basic language understanding.
            - BIG-bench currently consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more. BIG-bench comes in three different sizes, and part of the reason for this is to keep costs achievable, as running these large benchmarks can incur large inference costs.
        - HELM (By Stanford)
            
            ![image.png](Week%202%20Notes/image%2023.png)
            
            The HELM framework aims to improve the transparency of models, and to offer guidance on which models perform well for specific tasks. HELM takes a multimetric approach, measuring seven metrics across 16 core scenarios, ensuring that trade-offs between models and metrics are clearly exposed. 
            
            One important feature of HELM is that it assesses on metrics beyond basic accuracy measures, like precision of the F1 score. The benchmark also includes metrics for fairness, bias, and toxicity, which are becoming increasingly important to assess as LLMs become more capable of human-like language generation, and in turn of exhibiting potentially harmful behavior. 
            

---

## PEFT Intro
    - Full fine-tuning requires memory not just to store the model, but various other parameters that are required during the training process. Even if your computer can hold the model weights, which are now on the order of hundreds of gigabytes for the largest models, you must also be able to allocate memory for optimizer states, gradients, forward activations, and temporary memory throughout the training process. These additional components can be many times larger than the model and can quickly become too large to handle on consumer hardware.
        
        ![image.png](Week%202%20Notes/image%2024.png)
        
    - PEFT types (parameter efficient fine tuning methods only update a small subset of parameters. )
        - Some path techniques freeze most of the model weights and focus on fine tuning a subset of existing model parameters, for example, particular layers or components.
            
            ![image.png](Week%202%20Notes/image%2025.png)
            
        - Other techniques don't touch the original model weights at all, and instead add a small number of new parameters or layers and fine-tune only the new components.
            
            ![image.png](Week%202%20Notes/image%2026.png)
            
    - PEFT features
        - smaller memory requirement
            
            With PEFT, most if not all of the LLM weights are kept frozen. As a result, the number of trained parameters is much smaller than the number of parameters in the original LLM. In some cases, just 15-20% of the original LLM weights. In fact, PEFT can often be performed on a single GPU.
            
        - less prone to the catastrophic forgetting problems of full fine-tuning. because the original LLM is only slightly modified or left unchanged.
            - For multi-task fine tuning
                - Full fine-tuning results in a new version of the model for every task you train on. Each of these is the same size as the original model, so it can create an expensive storage problem.
                - With parameter efficient fine-tuning, you train only a small number of weights, which results in a much smaller footprint overall, as small as megabytes depending on the task. The new parameters are combined with the original LLM weights for inference.
                - The PEFT weights are trained for each task and can be easily swapped out for inference, allowing efficient adaptation of the original model to multiple tasks.
                
                ![image.png](Week%202%20Notes/image%2027.png)
                
                ![image.png](Week%202%20Notes/image%2028.png)
                
    - PEFT diff methods Trade-offs
        - There are several methods you can use for parameter efficient fine-tuning, each with trade-offs on parameter efficiency, memory efficiency, training speed, model quality, and inference costs.
            
            ![image.png](Week%202%20Notes/image%2029.png)
            
        - Three major types
            - Selective methods - Researchers have found that the performance of these methods is mixed and there are significant trade-offs between parameter efficiency and compute efficiency. We won't focus on them in this course.
                
                Selective methods are those that fine-tune only a subset of the original LLM parameters. There are several approaches that you can take to identify which parameters you want to update. You have the option to train only certain components of the model or specific layers, or even individual parameter types.
                
            - Reparameterization methods - Reduce the number of parameters to train by creating new low rank transformations of the original network weights.
                - A commonly used technique of this type is LoRA,
            - Addictive methods - Keeping all of the original LLM weights frozen and introducing new trainable components.
                - Two main approaches:
                    - Adapter methods
                        
                        add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers. 
                        
                    - Soft prompt methods
                        
                        keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights. (This course covers a specific soft prompts technique called prompt tuning later.)
                        
## PEFT techniques 1: LoRA (Low-rank Adaptation)
    - Recap of transformer structure - two kinds of neural networks(NN): Self-attention NN & Feed forward network
        - The input prompt is turned into tokens, which are then converted to embedding vectors and passed into the encoder and/or decoder parts of the transformer. In both of these components.
        - There are two kinds of neural networks; self-attention and feedforward networks. The weights of these networks are learned during pre-training.
        - After the embedding vectors are created, they're fed into the self-attention layers where a series of weights are applied to calculate the attention scores.
        
        ![image.png](Week%202%20Notes/image%2030.png)
        
        ![image.png](Week%202%20Notes/image%2031.png)
        
    - LoRA is a strategy that reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights.
        
        ![image.png](Week%202%20Notes/image%2032.png)
        
        - The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying. You then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process you saw earlier this week.
        - For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. You then add this to the original weights and replace them in the model with these updated values. Because this model has the same number of parameters as the original, there is little to no impact on inference latency.
        - Because LoRA allows you to significantly reduce the number of trainable parameters, you can often perform this method of parameter efficient fine tuning with a single GPU and avoid the need for a distributed cluster of GPUs.
    - Typically used on self-attention layer is good enough
        
        Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. However, in principle, you can also use LoRA on other components like the feed-forward layers. But since most of the parameters of LLMs are in the attention layers, you get the biggest savings in trainable parameters by applying LoRA to these weights matrices. 
        
    - LoRA example
        
        ![image.png](Week%202%20Notes/image%2033.png)
        
    - LoRA used to fine-tune a different set for each of the many tasks and switch each set of parameters at inference time by updating the according weights.
        
        ![image.png](Week%202%20Notes/image%2034.png)
        
        The memory required to store these LoRA matrices is very small. So in principle, you can use LoRA to train for many tasks. Switch out the weights when you need to use them, and avoid having to store multiple full-size versions of the LLM. 
        
    - LoRA performance comparing to full fine-tuning and base model using ROUGE scores (based on model FLAN-T5)
        
        ![image.png](Week%202%20Notes/image%2035.png)
        
        using LoRA for fine-tuning trained a much smaller number of parameters than full fine-tuning using significantly less compute, so this small trade-off in performance may well be worth it.
        
    - How to choose rank of the LoRA matrices - still an active research area
        
        ![image.png](Week%202%20Notes/image%2036.png)
        
        The authors found a plateau in the loss value for ranks greater than 16. In other words, using larger LoRA matrices didn't improve performance. The takeaway here is that ranks in the range of 4-32 can provide you with a good trade-off between reducing trainable parameters and preserving performance.
        
    - LoRA is a powerful fine-tuning method that achieves great performance. The principles behind the method are useful not just for training LLMs, but for models in other domains.
## PEFT techniques 2: Soft prompts (method: prompt tuning)
    - No changing to the weights at all
    - Prompt Engineering and Prompt Tuning are totally different
        - With prompt engineering, you work on the language of your prompt to get the completion you want. This could be as simple as trying different words or phrases or more complex, like including examples for one or Few-shot Inference. The goal is to help the model understand the nature of the task you're asking it to carry out and to generate a better completion.
            
            The limitation is that it requires a lot of manual effort to write and try. Also limited by the length of the context window.
            
    - With prompt tuning, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values.
        - The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent your input text.
        - The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere between 20 and 100 virtual tokens can be sufficient for good performance.
        
        ![image.png](Week%202%20Notes/image%2037.png)
        
        - The token represent natural language are dicrete, each corresponding to a fixed location in the embedding vector space. But soft prompts are not fixed discrete words of natural language. Instead, you can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space.
        
        ![image.png](Week%202%20Notes/image%2038.png)
        
        ![image.png](Week%202%20Notes/image%2039.png)
        
        - Through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task.
        
    - Full Fine-tuning vs Prompt tuning
        
        ![image.png](Week%202%20Notes/image%2040.png)
        
        In full fine tuning, the training data set consists of input prompts and output completions or labels. The weights of the large language model are updated during supervised learning. In contrast with prompt tuning, the weights of the large language model are frozen and the underlying model does not get updated. Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt.
        
    - Prompt tuning is a very parameter efficient strategy because only a few parameters are being trained.
    - Similar to LoRA,  easy to switch for multiple tasks
        
        ![image.png](Week%202%20Notes/image%2041.png)
        
        You can train a different set of soft prompts for each task and then easily swap them out at inference time. You can train a set of soft prompts for one task and a different set for another. To use them for inference, you prepend your input prompt with the learned tokens to switch to another task, you simply change the soft prompt. 
        
        Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible.
        
    - Prompt tuning performance
        
        ![image.png](Week%202%20Notes/image%2042.png)
        
        Prompt tuning doesn't perform as well as full fine tuning for smaller LLMs. However, as the model size increases, so does the performance of prompt tuning. And once models have around 10 billion parameters, prompt tuning can be as effective as full fine tuning and offers a significant boost in performance over prompt engineering alone. 
        
    - Potential issue: interpretability of learned virtual tokens
        
        The trained tokens don't correspond to any known token, word, or phrase in the vocabulary of the LLM. 
        
        ![image.png](Week%202%20Notes/image%2043.png)
        
        However, an analysis of the nearest neighbor tokens to the soft prompt location shows that they form tight semantic clusters. In other words, the words closest to the soft prompt tokens have similar meanings. The words identified usually have some meaning related to the task, suggesting that the prompts are learning word like representations.
        
## Summary of the two PEFT techs: Both methods enable you to fine tune models with the potential for improved performance on your tasks while using much less compute than full fine tuning methods. LoRA is broadly used in practice because of the comparable performance to full fine tuning for many tasks and data sets.

---

## Lab2 notes (https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/main/Week-2/Lab_2_fine_tune_generative_ai_model.ipynb)
    - Main goal of the lab is to compare original FLAN-T5 model, full fine-tuning approach, and PEFT using RoLA and evaluate the results with ROUGE metrics.
    - new libraries installed
        - evaluate (for ROUGE score)
        - loralib for LoRA
        - peft for PEFT
        - from transformer import TrainingArguments, Trainer
            - used to simplify the training of the model
    - installation and imports
        
        ```python
        %pip install --upgrade pip
        %pip install --disable-pip-version-check \
            torch==1.13.1 \
            torchdata==0.5.1 --quiet
        
        %pip install \
            transformers==4.27.2 \
            datasets==2.11.0 \
            evaluate==0.4.0 \
            rouge_score==0.1.2 \
            loralib==0.1.1 \
            peft==0.3.0 --quiet
          
        from datasets import load_dataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
        import torch
        import time
        import evaluate
        import pandas as pd
        import numpy as np
        ```
        
    - Get dataset and original model; Pull out number of model parameters that are trainable; Test the Model with Zero Shot Inferencing
        
        ```python
        
        huggingface_dataset_name = "knkarthick/dialogsum"
        
        dataset = load_dataset(huggingface_dataset_name)
        
        #--
        
        model_name='google/flan-t5-base'
        
        original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        #--
        
        def print_number_of_trainable_model_parameters(model):
            trainable_model_params = 0
            all_model_params = 0
            for _, param in model.named_parameters():
                all_model_params += param.numel()
                if param.requires_grad:
                    trainable_model_params += param.numel()
            return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
        
        print(print_number_of_trainable_model_parameters(original_model))
        ```
        
        ![image.png](Week%202%20Notes/image%2044.png)
        
        ![image.png](Week%202%20Notes/image%2045.png)
        
        ```python
        index = 200
        
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        
        prompt = f"""
        Summarize the following conversation.
        
        {dialogue}
        
        Summary:
        """
        
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
            original_model.generate(
                inputs["input_ids"], 
                max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
        )
        
        dash_line = '-'.join('' for x in range(100))
        print(dash_line)
        print(f'INPUT PROMPT:\n{prompt}')
        print(dash_line)
        print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
        print(dash_line)
        print(f'MODEL GENERATION - ZERO SHOT:\n{output}')
        ```
        
        ![image.png](Week%202%20Notes/image%2046.png)
        
    - Full Fine-Tuning - Preprocess Dataset with instructions; Fine-tuning with preprocess dataset; Downloaded a full version of fully fine-tuned model to save time; Evaluate the model using ROUGE score
        
        ```python
        def tokenize_function(example):
            start_prompt = 'Summarize the following conversation.\n\n'
            end_prompt = '\n\nSummary: '
            prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
            example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
            example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
            
            return example
        
        # The dataset actually contains 3 diff splits: train, validation, test.
        # The tokenize_function code is handling all data across all splits in batches.
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
        ```
        
        Now utilize the built-in Hugging Face `Trainer` class (see the documentation [here](https://huggingface.co/docs/transformers/main_classes/trainer)). Pass the preprocessed dataset with reference to the original model. Other training parameters are found experimentally and there is no need to go into details about those at the moment.
        
        ```python
        output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=1,
            max_steps=1
        )
        
        trainer = Trainer(
            model=original_model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation']
        )
        ```
        
        Training a fully fine-tuned version of the model would take a few hours on a GPU. To save time, download a checkpoint of the fully fine-tuned model to use in the rest of this notebook. This fully fine-tuned model will also be referred to as the **instruct model** in this lab.
        
        The size of the downloaded instruct model is approximately 1GB.
        
        ```python
        !aws s3 cp --recursive s3://dlai-generative-ai/models/flan-dialogue-summary-checkpoint/ ./flan-dialogue-summary-checkpoint/
        
        !ls -alh ./flan-dialogue-summary-checkpoint/pytorch_model.bin
        
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16)
        ```
        
        The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does indicate the overall increase in summarization effectiveness that we have accomplished by fine-tuning.
        
        ```python
        rouge = evaluate.load('rouge')
        
        #--
        
        dialogues = dataset['test'][0:10]['dialogue']
        human_baseline_summaries = dataset['test'][0:10]['summary']
        
        original_model_summaries = []
        instruct_model_summaries = []
        
        for _, dialogue in enumerate(dialogues):
            prompt = f"""
        Summarize the following conversation.
        
        {dialogue}
        
        Summary: """
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
            original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
            original_model_summaries.append(original_model_text_output)
        
            instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
            instruct_model_summaries.append(instruct_model_text_output)
            
         #--   
            
        original_model_results = rouge.compute(
            predictions=original_model_summaries,
            references=human_baseline_summaries[0:len(original_model_summaries)],
            use_aggregator=True,
            use_stemmer=True,
        )
        
        instruct_model_results = rouge.compute(
            predictions=instruct_model_summaries,
            references=human_baseline_summaries[0:len(instruct_model_summaries)],
            use_aggregator=True,
            use_stemmer=True,
        )
        
        print('ORIGINAL MODEL:')
        print(original_model_results)
        print('INSTRUCT MODEL:')
        print(instruct_model_results)
        ```
        
        ![image.png](Week%202%20Notes/image%2047.png)
        
    - PEFT - Setup LoRA model; Train PEFT Adapter; Downloaded a fully trained Adapter to save time; Evaluate using ROUGE.
        - In most cases, when someone says PEFT, they typically mean LoRA. After fine-tuning for a specific task, use case, or tenant with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges. This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).
        - That said, at inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request. The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.
        
        ```python
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=32, # Rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )
        
        # Add LoRA adapter layers/parameters to the original LLM to be trained.
        
        peft_model = get_peft_model(original_model, 
                                    lora_config)
        print(print_number_of_trainable_model_parameters(peft_model))
        ```
        
        ![image.png](Week%202%20Notes/image%2048.png)
        
        ```python
        output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
        
        peft_training_args = TrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3, # Higher learning rate than full fine-tuning.
            num_train_epochs=1,
            logging_steps=1,
            max_steps=1    
        )
            
        peft_trainer = Trainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=tokenized_datasets["train"],
        )
        
        #--
        
        peft_trainer.train()
        
        peft_model_path="./peft-dialogue-summary-checkpoint-local"
        
        peft_trainer.model.save_pretrained(peft_model_path)
        tokenizer.save_pretrained(peft_model_path)
        
        # That training was performed on a subset of data. To load a fully trained PEFT model, read a checkpoint of a PEFT model from S3.
        
        !aws s3 cp --recursive s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint-from-s3/ 
        ```
        
        Prepare this model by adding an adapter to the original FLAN-T5 model. You are setting `is_trainable=False` because the plan is only to perform inference with this PEFT model. If you were preparing the model for further training, you would set `is_trainable=True`.
        
        ```python
        from peft import PeftModel, PeftConfig
        
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        peft_model = PeftModel.from_pretrained(peft_model_base, './peft-dialogue-summary-checkpoint-from-s3/', torch_dtype=torch.bfloat16, is_trainable=False)
        
        #--
        
        # The number of trainable parameters will be 0 due to is_trainable=False setting:
        
        print(print_number_of_trainable_model_parameters(peft_model))
                                              
                           
        ```
        
        ```python
        # Evaluate (Here with a very small test set due to time limits)
        dialogues = dataset['test'][0:10]['dialogue']
        human_baseline_summaries = dataset['test'][0:10]['summary']
        
        original_model_summaries = []
        instruct_model_summaries = []
        peft_model_summaries = []
        
        for idx, dialogue in enumerate(dialogues):
            prompt = f"""
        Summarize the following conversation.
        
        {dialogue}
        
        Summary: """
            
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
            human_baseline_text_output = human_baseline_summaries[idx]
            
            original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        
            instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
        
            peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        
            original_model_summaries.append(original_model_text_output)
            instruct_model_summaries.append(instruct_model_text_output)
            peft_model_summaries.append(peft_model_text_output)
        
        zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))
         
        df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries', 'peft_model_summaries'])
        df
        ```
        
        ```python
        rouge = evaluate.load('rouge')
        
        original_model_results = rouge.compute(
            predictions=original_model_summaries,
            references=human_baseline_summaries[0:len(original_model_summaries)],
            use_aggregator=True,
            use_stemmer=True,
        )
        
        instruct_model_results = rouge.compute(
            predictions=instruct_model_summaries,
            references=human_baseline_summaries[0:len(instruct_model_summaries)],
            use_aggregator=True,
            use_stemmer=True,
        )
        
        peft_model_results = rouge.compute(
            predictions=peft_model_summaries,
            references=human_baseline_summaries[0:len(peft_model_summaries)],
            use_aggregator=True,
            use_stemmer=True,
        )
        
        print('ORIGINAL MODEL:')
        print(original_model_results)
        print('INSTRUCT MODEL:')
        print(instruct_model_results)
        print('PEFT MODEL:')
        print(peft_model_results)
        ```
        
        ![image.png](Week%202%20Notes/image%2049.png)
        
## [Week2 Resources](https://www.coursera.org/learn/generative-ai-with-llms/supplement/zlpBf/week-2-resources)