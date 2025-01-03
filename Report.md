# slop-radar: An attempt in identifying human and AI-generated texts

## Abstract
// Summarise every other section
## Introduction
*This section places your project in a machine learning context as exemplified by the academic literature. You can amplify the aim as stated in the abstract and explain why this aim is interesting and relevant. You might introduce your dataset(s) and explain any particular problems both with the dataset and with known investigations of this dataset.*

When OpenAI released [ChatGPT](https://openai.com/index/chatgpt/) in November of 2022, it disrupted multiple industries and many professionals considered the ways they could use it to improve their productivity. Unfortunately,  instead of viewing it as just another tool, many users have taken this to mean replacing entire tasks, like writing emails or drafting documents from scratch. Our current state-of-the-art *Large Language Models* suffer from a high rate of mistakes and an inability to fact check. People find their messages distant and impersonal as they lack the context that a human would have. LLMs have seen widespread use in an academic setting, and have had an equally widespread pushback as a form of academic dishonesty. Traditional plagiarism-checking services like Turnitin have even started to offer [detection of AI-written work](https://www.turnitin.com/blog/the-launch-of-turnitins-ai-writing-detector-and-the-road-ahead). Their ability to write in a convincing and authoritative manner with minimal effort has greatly simplified the creation of disinformation campaigns and scams. [(Kumarage et al, 2024)](#^1)

Just as literacy has become necessary for all after the industrial age, I believe that digital literacy is necessary for all to stay safe in the near future. I consider myself an early adopter as I started to [download and run such models](https://huggingface.co/cognitivecomputations/WizardLM-33B-V1.0-Uncensored) off my own computer in July of 2023. [Enthusiasts in the LLM space](https://www.reddit.com/r/LocalLLaMA/comments/17xwuno/what_do_you_think_about_gptisms_polluting/) who have seen enough generated text know their shortcomings more intimately and have developed a better sense of whether any text is written by a human. They try to identify what they call *GPT-isms* or *slop*: choice phrases and figures of speech that are stereotypical of ChatGPT or other models, which can be useful for a layperson to detect AI-generated text themselves.

// Find literature for me that aligns with my observation "They focus on the effectiveness of detection, but they do not seem to draw conclusions that the general public can follow, and it may be impractical to rely on detection algorithms at all times."

In this project, I prepare and analyse a corpus of texts written by a roster of models, and aim to identify choice vocabulary that distinguishes between human and AI-generated texts. I also implement and evaluate the abilities of the *k-nearest neighbours* algorithm to identify the exact author of any text. Finally, I conclude with some actionable steps to identify AI-generated text.
## Background
In this section, I describe the building blocks of the project: the dataset and the algorithms that were used.
### Dataset
[*Chatbot Arena*](https://chat.lmsys.org/) is an online platform that hosts multiple LLMs. When a user starts a conversation, the platform automatically chooses two of its models to provide responses. The user may continue the conversation with both anonymous models until they are ready to decide on the better model. In addition to developing Chatbot Arena, the team also released a paper that evaluated the strengths of their models over an 8-month period, as well as a curated dataset of conversations on the platform [(Chiang et al, 2024)](#^2).

Provided that you have accepted the dataset's terms of service on Hugging Face (which you can do by [following the link](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)), the dataset can be downloaded and used immediately in my accompanying notebook, by entering an access token associated with your Hugging Face account when prompted.

|     | question_id                      | model_a    | model_b    | winner  | judge          | conversation_a                                    | conversation_b                                    | turn | anony | language | tstamp       | openai_moderation                                 | toxic_chat_tag                                    |
| --- | -------------------------------- | ---------- | ---------- | ------- | -------------- | ------------------------------------------------- | ------------------------------------------------- | ---- | ----- | -------- | ------------ | ------------------------------------------------- | ------------------------------------------------- |
| 33  | 8120899314f74641b09c2aa114d4d253 | alpaca-13b | vicuna-13b | model_b | arena_user_316 | [{'content': 'Salut ! Comment ça va ce matin ?... | [{'content': 'Salut ! Comment ça va ce matin ?... | 6    | True  | French   | 1.682354e+09 | {'categories': {'harassment': False, 'harassme... | {'roberta-large': {'flagged': False, 'probabil... |

This is a typical row in the dataset. Every entry describes the metadata of a conversation between a user and two models not known to the user. The actual conversation has to be unpacked from `conversation_a` and `conversation_b`.

```
[{'content': 'Salut ! Comment ça va ce matin ?', 'role': 'user'},
 {'content': 'Ça va bien, merci. Et toi ?', 'role': 'assistant'},
 {'content': 'Ça va et toi ?', 'role': 'user'},
 {'content': 'Oui, et toi ?', 'role': 'assistant'},
 {'content': 'Ça va et vous ?', 'role': 'user'},
 {'content': 'Oui, et vous ?', 'role': 'assistant'},
 {'content': 'ça va', 'role': 'user'},
 {'content': 'ça va', 'role': 'assistant'},
 {'content': 'non ça va', 'role': 'user'},
 {'content': 'non ça va', 'role': 'assistant'},
 {'content': 'Ok ', 'role': 'user'},
 {'content': 'Ok', 'role': 'assistant'}]
```

This particular `conversation_a` shows that the user and the two models shared 6 messages each throughout their conversation. Messages with the `role` of `assistant` were written by `model_a`, which in this case was *alpaca-13b*.

```
[{'content': 'Salut ! Comment ça va ce matin ?', 'role': 'user'},
 {'content': "Salut ! Je vais bien, merci. Comment allez-vous aujourd'hui ?",
  'role': 'assistant'},
 {'content': 'Ça va et toi ?', 'role': 'user'},
 {'content': 'Oui, je vais bien également. Comment ça se passe pour vous ?',
  'role': 'assistant'},
 {'content': 'Ça va et vous ?', 'role': 'user'},
 {'content': "Oui, je vais bien merci. Comment ça se passe pour vous aujourd'hui ?",
  'role': 'assistant'},
 {'content': 'ça va', 'role': 'user'},
 {'content': "C'est super. Y a-t-il quelque chose que je puisse faire pour vous aider aujourd'hui ?",
  'role': 'assistant'},
 {'content': 'non ça va', 'role': 'user'},
 {'content': "Très bien, si vous avez besoin de quelque chose, n'hésitez pas à me demander. Je suis là pour vous aider.",
  'role': 'assistant'},
 {'content': 'Ok ', 'role': 'user'},
 {'content': "Très bien. N'hésitez pas à me contacter si vous avez besoin de quelque chose. Je suis toujours là pour vous aider.",
  'role': 'assistant'}]
```

Messages with the `role` of `assistant` in `conversation_b` were written by `model_b`, which in this case was *vicuna-13b*. Here, we can also see that messages with the `role` of `user` are identical across both conversations. They were written by the `judge`, *arena_user_316*, which is an anonymised identifier that Chatbot Arena has given the user. From the platform's perspective, users are differentiated by their IP addresses, but since IP addresses and devices can be shared, it is not given that messages from the same user are written by the same person.

The other fields are helpful in the event that the dataset is used to rank models against each other (using `winner`) or train a classifier to differentiate other attributes (`language`, `openai_moderation`, or `toxic_chat_tag`). However, for the purpose of identifying trends in model vocabulary, these fields have no use and I shall not describe them here.
### Algorithms
Here, I give an overview of the algorithms that were used, as well as the context in which I used them.
#### [`sklearn.feature_extraction.text.CountVectorizer.fit_transform()`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform) from `scikit-learn`
By passing in a corpus of texts, this algorithm provides vectors of *token* counts for each text, where each token roughly corresponds to a word. With default parameters, it derives a list of tokens by separating texts at spaces, and removes duplicates and predetermined stop words like "the" and "a" from the list. The resulting list is the *vocabulary*, which is a list of all unique tokens that appear at least once across the corpus. Then, for each text, the algorithm counts the number of times each token in the vocabulary appears. It returns a two-dimensional array of text indices on one side and the list of tokens on the other.

For a large enough corpus, this usually results in a sparse matrix, as most tokens in the vocabulary don't appear at least once in any given text, resulting in many zeroes. This algorithm is used in the notebook to provide a count of word frequency per text in the dataset, as numerical features are more readily learned than raw text.
#### Principal Component Analysis
Given a set of points, this algorithm finds a new coordinate system where the first few axes (principal components) capture the directions of maximum variance. In other words, if the points are projected into this system and some dimensions further back are dropped, much of the relationship between the points (variance) is maintained.

The process of finding principal components is iterative and an intuitive method is often taught, where a line is fit through the centre, so that the perpendicular distance between the points and the line is minimised. *Gradient descent* with an appropriate learning rate (alpha) can gradually align the line in the right direction. However, this fitting can be replaced by *eigenvalue algorithms*. By finding the *covariance matrix* (the relationship between each and every other point under a constant change), I can find its *eigenvector* using *power iteration*, which I shall explain below.

PCA makes further processing easier, especially for other algorithms whose time and memory footprint disproportionately increases with more dimensions. However, a reduced principal component space loses some of the variance. Therefore, it is useful to have a *hyperparameter* that determines how many principal components we would like to keep. Higher values result in more retained components, making the approximation of the points more accurate but nullifying the dimensionality reduction advantage of PCA. Meanwhile, lower values speed up processing but risk discarding relevant patterns.

In this project, where the points represent batches of texts from each author, I use PCA to reduce the dimensionality of token frequency vectors while retaining the relationships that can help indicate authorship (variance). This accelerates further operations such as k-nearest neighbours. Additionally, the first few principal components also surface the most important tokens when discriminating between authors.
#### Power iteration
By passing in a matrix, this algorithm provides its *dominant eigenvalue* and corresponding *eigenvector*. In my implementation, it starts with a random vector, multiplies that vector by the matrix and normalises the result. Provided that the starting vector isn't orthogonal to the eigenvector, it will stretch/squeeze to be more and more like the eigenvector. I repeat this multiplication repeatedly until the vector changes too little to matter — convergence. I define an error tolerance, below which the iteration ends.

The goal of each step in PCA is to find the next direction with the largest spread of values. In that context, the eigenvector of the covariance matrix is equivalent to that direction, and the eigenvalue to the explained variance. There are many other eigenvalue algorithms, but power iteration was the easiest for me to wrap my head around. It is also memory-efficient, which is ideal for the memory footprint of sparse matrices. This simplicity comes as a trade-off to its slow convergence compared to more advanced algorithms like the *QR decomposition*.
#### k-nearest neighbours
The definition is in its name: it plots an unlabelled testing point against a labelled set of *training* points, and predicts a label based on a plurality of `k` of its nearest neighbours. Since a training set with labels has to already be provided, this makes k-NN the only supervised learning algorithm in this section.

Due to its simplicity many variations exist, and the available hyperparameters can vary by implementation. For example, "near" can depend on the [distance metric](https://uhurasolutions.com/2020/07/15/a-short-introduction-to-k-nearest-neighbor-or-simply-k-nn/) being used. Commonly, the Euclidean distance is used with numerical and continuous values, although different methods may yield better results (such as Hamming/Levenshtein distance for text or Manhattan distance for discrete values). One could even implement multiple metrics to find the best. In the case of ties, the labels of closer neighbours can be weighted more, or the test point can be resampled with the original neighbours removed: *bootstrapping*.

In my implementation, I've decided to keep it simple, Euclidean distance being the distance metric and a decreasing `k` for tiebreaks. This results in one hyperparameter: `k`. Larger values capture trends in the big picture but may perform poorly for label clusters that are close to each other. Lower values may provide that precision but may risk overfitting to outlier training points. In this project, k-NN is used to predict the author of a batch of text projected on a reduced PC space. It helps measure the extent to which the first few PCs can differentiate between human and AI-generated texts.
## Methodology
// In this section I ...
### Preparing the dataset
In this section, I describe how I transformed multiple messages from each row of the dataset into a structured format, suitable for analysis.
#### Extracting text-author pairs
From each row in the dataset I identified the following 3 kinds of texts:
- Messages from `model_a`, found by filtering entries in `conversation_a` where the `role` is `assistant`.
- Messages from `model_b`, found by filtering entries in `conversation_b` where the `role` is `assistant`. Since both models respond alternately to the same prompts, the number of messages here matches the number of messages from `model_a`.
- Messages from a human, found by filtering entries in either conversation where the `role` is `user`. The author would be the blanket term "human".

I considered further differentiating human texts by their `judge` identifiers (e.g., `arena_user_316`). However, apart from a few power users there weren't enough text from each author to form a single batch of 200 texts. Normalising a smaller batch means that any token that exists is overrepresented. Additionally, as discussed above, there is no guarantee that the same user consistently represents a single human.

```
vicuna-13b: 7193
koala-13b: 6615
oasst-pythia-12b: 5774
gpt-3.5-turbo: 5573
alpaca-13b: 5399
gpt-4: 4945
claude-v1: 4500
RWKV-4-Raven-14B: 4392
chatglm-6b: 3791
fastchat-t5-3b: 3699
vicuna-7b: 3578
palm-2: 3527
mpt-7b-chat: 3516
dolly-v2-12b: 3307
stablelm-tuned-alpha-7b: 3249
claude-instant-v1: 3117
llama-13b: 2393
gpt4all-13b-snoozy: 1425
wizardlm-13b: 1391
guanaco-33b: 1248
arena_user_15085: 458
arena_user_9965: 434
arena_user_13046: 385
arena_user_257: 296
arena_user_11473: 240
arena_user_3820: 213
arena_user_9676: 166
52 authors who have between 50 and 138 texts.
91 authors who have between 25 and 49 texts.
172 authors who have between 15 and 24 texts.
252 authors who have between 10 and 14 texts.
619 authors who have between 6 and 9 texts.
1877 authors who have between 3 and 5 texts.
10313 authors who have between 1 and 2 texts.
```

Instead, I rolled all 39316 human texts into a single author: "human". To reduce the variance between batches and better approximate the average writing style of this human and each model, I shuffled all extracted text-author pairs before batching. As the messages in each conversation were spread across the batches, words that are common only in niche topics were also homogenised.
#### Creating token count vectors for each text
As discussed earlier, I used `CountVectorizer.fit_transform()` with its default settings to convert the dataset into numerical features. With the default parameters I used, this algorithm applied stop word removal based on its built-in list for the English language. `scikit-learn` also has built-in lists for other languages that I can specify to use in the `stop_words` parameter. In hindsight, some conversations in the dataset were identified to be in another language, and to prevent their stop words from being overrepresented it may have been appropriate to remove them too.

```
Out of 117948 messages:
English: 103569
German: 2400
Spanish: 2184
French: 1626
unknown: 1377
Portuguese: 1041
Russian: 990
Italian: 486
Chinese: 480
Dutch: 306
Polish: 300
Japanese: 297
Finnish: 183
Korean: 174
Danish: 168
Latin: 132
Vietnamese: 117
Turkish: 117
Indonesian: 111
Czech: 108
Ukrainian: 105
Scots: 99
Swedish: 87
Slovak: 84
Hebrew: 81
Romanian: 75
Arabic: 69
Hungarian: 69
Norwegian: 60
Slovenian: 57
Persian: 57
Galician: 54
Greek: 48
Esperanto: 45
Thai: 42
Catalan: 39
Luxembourgish: 39
Waray: 33
Hawaiian: 30
Bislama: 30
Uzbek: 30
Afrikaans: 27
Bulgarian: 27
Norwegian Nynorsk: 27
Tsonga: 24
Maltese: 21
Icelandic: 21
Serbian: 21
Malay: 18
Quechua: 15
Ganda: 15
Southern Sotho: 15
Corsican: 15
Tigrinya: 15
Xhosa: 12
Akan: 12
Interlingua: 12
Kinyarwanda: 12
Sanskrit: 12
Macedonian: 12
Faroese: 12
Estonian: 12
Haitian Creole: 9
Fijian: 9
Aymara: 9
Latvian: 9
Afar: 9
Welsh: 9
Tswana: 9
Hindi: 9
Maori: 9
Occitan: 9
Somali: 6
Oromo: 6
Wolof: 6
Volapük: 6
Swahili: 6
Croatian: 6
Shona: 6
zzp: 6
Malagasy: 6
Hmong: 6
Manx: 3
Lithuanian: 3
Sundanese: 3
Klingon: 3
Cebuano: 3
Tatar: 3
Kalaallisut: 3
Irish: 3
Basque: 3
Bangla: 3
Interlingue: 3
Western Frisian: 3
Yoruba: 3
Guarani: 3
```

However, retroactively looking at the amount of messages attributed to each language, 88% of all texts were in English, and German was the next most represented language at just 2% of all texts. Given the nature of my batching process to dilute texts of other languages among English ones (i.e., 4 German texts to 176 English texts in an average batch), their stop words might be less frequent than most other English words, let alone English stop words. My aggressive feature selection process further reduced the presence of non-English tokens, such that in any run I was unable to spot any token that wasn't English being retained. Therefore, I consider this a non-issue.

I could also consider filtering out non-English texts. Since the intended audience of this project are English speakers, the tokens from other texts could have been considered as noise. However, if models that don't exhibit distinctive patterns in English, show a difference in other languages instead, those are still valuable insights that are actionable (e.g., force it to speak Spanish and look out for certain keywords). Though likely undetectable in this project, features like verbosity and repetitiveness may be language-agnostic and are worth preserving for analysis.
#### Grouping into batches
A batch represents approximately 200 texts attributed to a single author. I had arbitrarily chosen a size of 200 after comparing the amount of texts from each author; this meant that the least represented model, *guanaco-33b*, would still have 6 batches (6 sample points). Each text is represented by a vector of token counts across the vocabulary space, and each batch is computed as the sum of all vectors within it.

First, I calculated the number of batches required for each author by dividing the total number of texts by 200, rounding as needed. Then, I calculated the quota for each batch by distributing the remainder or shortfall from rounding evenly across batches. Finally, I added the text-author pairs that had been shuffled earlier to the next available batch for each author until the batch size quota was reached.

| |0_human|0_claude-instant-v1|0_RWKV-4-Raven-14B|0_oasst-pythia-12b|0_chatglm-6b|0_palm-2|0_dolly-v2-12b|0_koala-13b|0_wizardlm-13b|0_alpaca-13b|...|190_human|28_oasst-pythia-12b|191_human|32_koala-13b|192_human|35_vicuna-13b|193_human|194_human|195_human|196_human|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|00|1|0|0|2|0|7|0|2|3|0|...|2|0|0|0|1|0|0|10|1|50|
|000|0|12|4|6|0|8|1|6|5|3|...|11|2|0|8|0|5|3|0|0|0|
|0000|0|0|0|0|0|1|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|00000|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|000000|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
|ｔｏ|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|ｗｅｅｋｓ|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|𝘀𝗶𝗺𝗽𝗹𝗲|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|𝘀𝘂𝗽𝗲𝗿𝗵𝘂𝗺𝗮𝗻|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|
|𝘀𝘂𝗽𝗲𝗿𝗵𝘂𝗺𝗮𝗻𝘀|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|0|0|0|0|

195411 rows × 591 columns

As a form of normalisation, I extracted the length of the average text in each batch as an additional feature. Then, I divided the token counts by this length so that their sum equalled **1**. Now, the batches represented token frequencies instead.

|                        | 0_human   | 0_claude-instant-v1 | 0_RWKV-4-Raven-14B | 0_oasst-pythia-12b | 0_chatglm-6b | 0_palm-2   | 0_dolly-v2-12b | 0_koala-13b | 0_wizardlm-13b | 0_alpaca-13b | ... | 190_human | 28_oasst-pythia-12b | 191_human | 32_koala-13b | 192_human | 35_vicuna-13b | 193_human | 194_human | 195_human | 196_human |
| ---------------------- | --------- | ------------------- | ------------------ | ------------------ | ------------ | ---------- | -------------- | ----------- | -------------- | ------------ | --- | --------- | ------------------- | --------- | ------------ | --------- | ------------- | --------- | --------- | --------- | --------- |
| 00                     | 0.000184  | 0.000000            | 0.000000           | 0.000102           | 0.000000     | 0.000252   | 0.000000       | 0.000071    | 0.000125       | 0.000000     | ... | 0.000289  | 0.000000            | 0.000     | 0.000000     | 0.000197  | 0.000000      | 0.000000  | 0.001566  | 0.000163  | 0.00959   |
| 000                    | 0.000000  | 0.000431            | 0.000191           | 0.000306           | 0.000000     | 0.000288   | 0.000088       | 0.000214    | 0.000208       | 0.000322     | ... | 0.001589  | 0.000109            | 0.000     | 0.000301     | 0.000000  | 0.000197      | 0.000513  | 0.000000  | 0.000000  | 0.00000   |
| 0000                   | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000036   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| 00000                  | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000000   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| 000000                 | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000000   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| ...                    | ...       | ...                 | ...                | ...                | ...          | ...        | ...            | ...         | ...            | ...          | ... | ...       | ...                 | ...       | ...          | ...       | ...           | ...       | ...       | ...       | ...       |
| ｗｅｅｋｓ                  | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000000   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| 𝘀𝗶𝗺𝗽𝗹𝗲           | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000000   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| 𝘀𝘂𝗽𝗲𝗿𝗵𝘂𝗺𝗮𝗻   | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000000   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| 𝘀𝘂𝗽𝗲𝗿𝗵𝘂𝗺𝗮𝗻𝘀 | 0.000000  | 0.000000            | 0.000000           | 0.000000           | 0.000000     | 0.000000   | 0.000000       | 0.000000    | 0.000000       | 0.000000     | ... | 0.000000  | 0.000000            | 0.000     | 0.000000     | 0.000000  | 0.000000      | 0.000000  | 0.000000  | 0.000000  | 0.00000   |
| length                 | 27.331658 | 143.654639          | 105.507538         | 98.557789          | 143.452261   | 142.266667 | 58.268041      | 140.160000  | 121.368687     | 46.783920    | ... | 34.605000 | 91.770000           | 24.455    | 132.099502   | 25.447236 | 126.740000    | 29.245000 | 32.095477 | 30.605000 | 26.07000  |

195412 rows × 591 columns

At this point, the dataset was ready for analysis. It had been grouped into batches, attributed to any of the models or the average human. Each batch was a vector of token frequencies across the entire vocabulary space. From here, the batches would serve as the input units for multiple runs of the subsequent analysis: feature extraction, PCA, and k-NN.
### Visualising a single run
In this section, I describe the steps of my analysis using a single run as an example.
#### Custom holdout
One batch per author was chosen at random to be holdout data. This means that the holdout dataset for the run consisted of one batch from each author (21 batches). Consequently, the 570 remaining batches were designated to be training data. In this particular run, I seeded `random` with `0`.

I had devised this form of holdout after reviewing some common schemes of cross-validation, and considering the circumstances of my dataset. Based on the environment in which I intended to run the notebook, I estimated that any *exhaustive* method of cross-validation would be too time-consuming. Then, due to the large number of unique labels and the low population of some labels, a run of *k-fold validation* may, at random, cause a majority of a label's population to become part of a fold.

By taking only one batch from each label, this scheme gives every label in the training dataset the highest amount of representation possible, yet ensures that every label is tested at least once. The holdout is not *stratified*, as it does not approximate the proportion of authors in the dataset as a whole. I also intended to run the analysis multiple times, using different seeds when separating the holdout. Since different batches would be chosen for each run, this scheme would also display characteristics of *Monte-Carlo* cross-validation, where different permutations of holdout batches are randomly explored to approximate the overall performance.
#### Feature selection
There were 195411 unique tokens in the vocabulary and an additional text length feature. Even as a dimensionality reduction algorithm, running PCA in a 195412-dimensional space would be computationally unfeasible. Therefore, I dropped tokens that did not appear at least once across all 374 non-human training batches, then dropped the same tokens from the human training batches and holdout batches.

```
Of 195412 tokens:
177337 of them appear at least once in 1 or more LLM batches.
77863 of them appear at least once in 2 or more LLM batches.
42947 of them appear at least once in 4 or more LLM batches.
25914 of them appear at least once in 8 or more LLM batches.
16194 of them appear at least once in 16 or more LLM batches.
10155 of them appear at least once in 32 or more LLM batches.
6169 of them appear at least once in 64 or more LLM batches.
3423 of them appear at least once in 128 or more LLM batches.
1367 of them appear at least once in 256 or more LLM batches.
132 of them appear at least once in all 374 LLM batches.
```

Manually dropping features came at the risk of losing insightful data, but I arrived at this seemingly arbitrary decision through experimentation. First, in order to accentuate the differences between individual models, I decided that tokens unique to only the human author should be dropped. After all, comparing how frequently each model used a word which only a human used would provide no information. Then, to avoid measuring differences in word frequency when only one model would use that word, I excluded tokens that only appeared in one batch.

I still had too many dimensions to work with — over a third of the original amount, so I tried to exclude tokens that did not appear in at least half of all batches, with the reasoning that any dimension where more than half of observations were zero essentially had more noise than signal. I arrived at a satisfactory number of dimensions in the few thousands, but I found that even if I excluded all tokens that did not appear at least once in all batches, there were still enough dimensions — 132 in this particular run, to justify reducing with PCA.

Following the aggressive feature selection where less than 0.1% of original features were retained, all batches used the same inventory of common words, but at varying, non-zero frequencies. I believe this made comparisons between the writing styles of each model noise-free and of significantly higher quality. Since human-specific tokens were dropped, the human and holdout batches could then be compared to these models solely in the context of their words.
#### Centring and standardising
```py
def centre(dataset, holdout_dataset) :
    features_mean = pd.Series({feature: np.mean(dataset.loc[feature]) for feature in dataset.index})
    for batch in dataset.columns :
        dataset[batch] -= features_mean
    for batch in holdout_dataset.columns :
        holdout_dataset[batch] -= features_mean
    return dataset, holdout_dataset

def standardise(dataset, holdout_dataset) :
    features_std = pd.Series({feature: np.std(dataset.loc[feature]) for feature in dataset.index})
    for batch in dataset.columns :
        dataset[batch] /= features_std
    for batch in holdout_dataset.columns :
        holdout_dataset[batch] /= features_std
    return dataset, holdout_dataset
```

Following normalization, token frequencies were standardized feature-wise (z-scores) to address differences in scale and magnitude. Each feature's mean was subtracted, and values were divided by standard deviation, aligning the data for PCA and distance-based methods like k-NN. The holdout batches underwent the same transformations using the training set parameters to preserve independence.
#### PCA with power iteration
#### k-nearest neighbours
### Runs across hyperparameters
#### Testing regime
#### Confusion matrix and F-score
## Results
### Visualising a single evaluation
### Confusion matrix and F-score
| d   | k   | Micro F_1 score |
| --- | --- | --------------- |
| 26  | 1   | 0.592063        |
| 26  | 2   | 0.592063        |
| 31  | 9   | 0.592063        |
| 26  | 3   | 0.590476        |
| 31  | 10  | 0.588889        |
| ... | ... | ...             |
| 1   | 5   | 0.195238        |
| 1   | 6   | 0.195238        |
| 1   | 1   | 0.193651        |
| 1   | 2   | 0.193651        |
| 1   | 3   | 0.193651        |

750 rows × 3 columns
## Evaluation
*This is a chance to demonstrate a critical awareness of the strengths and weaknesses of your project. Remember that a research project is judged by referring to the stated aim. A project does not have to succeed! Marks are allocated for how you undertook the project and for understanding and insight. A very ambitious aim might be unachievable within the terms of this project.*

conclusions from single evaluation (to tell apart human from synthetic writing)
conclusions from testing run (to tell apart writing styles)

## Conclusions
*State succinctly your findings and how they relate to your aim.*
## References
*List any academic work (such as a book or a research paper) that you refer to in the main body of the report.*
1. https://doi.org/10.48550/arXiv.2403.01152 ^1
2. https://doi.org/10.48550/arXiv.2403.04132 ^2