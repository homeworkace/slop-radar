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
In this section, I describe the steps of my analysis using a single run as an example. While developing the functions in this notebook, I used this run to iteratively test and improve the flow of data for the multiple runs to come. I also regularly paused to add visualisations during the run to understand the characteristics of the data and justify some design decisions. 
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

Before performing PCA, it's a good practice to normalise the data points. Without centralisation, the first principal component is the direction to the mean, rather than the direction of maximum variance. Without scaling, the *loading* (contribution from each dimension) for the first few PCs will be dominated by dimensions with observations across a larger magnitude, making human interpretation inaccurate.

First, I subtracted the mean from each training batch to centre the data. Then, I divided each dimension by their standard deviations to standardise the data. As a result, the batches had a mean of 0, and a standard deviation of 1 for each of its 132 dimensions in this run. Each dimension of the dataset contributed 1 unit of variance, for a total variance of 132. Finally, I applied the same transformation to the holdout batches. The holdout batches were not factored in when computing the means and standard deviations to simulate the projection of unseen data onto the learned space. In hindsight, my functions could have returned the means and standard deviations along with the normalised data, so that I could project new data into transformed space or existing data back into the original space. 
#### PCA with power iteration
```py
def power_iteration(A, num_iter, tol, single_run_diagnostics = False):
    """
    Computes the dominant eigenvalue and corresponding eigenvector of matrix A.
    Args:
        A (np.ndarray): Input symmetric matrix.
        num_iter (int): Number of iterations.
        tol (float): Tolerance for convergence.
        single_run_diagnostics (bool): If True, shows a progress bar to convergence.

    Returns:
        (np.ndarray): Approximation of the largest eigenvector.
        (float): Corresponding eigenvalue.
    """
    # Start with a random unit vector
    b = np.random.rand(A.shape[1])
    b = b / np.linalg.norm(b)

    # Start a progress bar
    if single_run_diagnostics :
        error_meter = tqdm(position = 0, total = 7, bar_format='{bar} | {postfix}')
        
    for i in range(num_iter):
        b_next = A @ b # Multiply, and...
        b_next = b_next / np.linalg.norm(b_next) # ... normalise!
        
        # Check for convergence
        error = np.linalg.norm(b_next - b)
        b = b_next
        if single_run_diagnostics :
            error_meter.update(min(-np.log10(error), error_meter.total) - error_meter.n)
            error_meter.set_postfix_str('{:.2E}'.format(error) + ' of error after ' + str(i + 1) + ' iterations...')
        if error < tol:
            if single_run_diagnostics :
                error_meter.close()
            break

    eigenvalue = b.T @ A @ b  # Rayleigh quotient
    return b, eigenvalue

def pca_power_iteration(data, num_components, num_iter, single_run_diagnostics = False):
    """
    Principal component analysis with power iteration to compute the principal components.
    Args:
        data (np.ndarray): Input data matrix (rows are samples, columns are features).
        num_components (int): Number of principal components to compute.
        num_iter (int): Number of iterations for power iteration.
        single_run_diagnostics (bool): Passes value to power_iteration(). If True, prints whenever a new component is being calculated.

    Returns:
        (list): List of principal components (eigenvectors).
        (list): List of corresponding eigenvalues.
    """
    components = []
    explained_variance = []
    
    # Compute covariance matrix
    n = data.shape[1]
    covariance_matrix = (data @ data.T) / n

    for i in range(num_components):
        # Compute the largest eigenvector using power iteration
        if single_run_diagnostics :
            print('Computing PC' + str(i + 1) + '...')
        eigenvector, eigenvalue = power_iteration(covariance_matrix, num_iter, 1e-7, single_run_diagnostics)
        components.append(eigenvector)
        explained_variance.append(eigenvalue)
        
        # Deflate the covariance matrix
        covariance_matrix -= eigenvalue * np.outer(eigenvector, eigenvector)
        if np.linalg.norm(covariance_matrix, ord = 'fro') < 1e-10 :
            if single_run_diagnostics :
                print('Stopped as all variance has been extracted after computing ' + str(i + 1) + ' PCs.')
            break
        
    return components, explained_variance

def transform_to_pc(batch, components) :
    pc_space_batch = []
    for component in components :
        pc_space_batch.append(np.dot(batch, component))
        
    return pc_space_batch
```

I implemented PCA and power iteration only with the help of `numpy` operations. While testing them, I also added progress bars to ensure that my algorithms were working, and that they weren't stuck in loops.
Finally, I created a utility function to project all batches onto PC space, by passing in as many PCs as I wanted. I assembled the testing labels with the fact that the holdout dataset is constructed by iterating through the list of unique authors, whose order stayed the same regardless of runs.

In this particular run I computed 50 principal components. Between 374 training batches and 132 dimensions there would be up to 132 PCs. I intended to have an excess of PCs to work with, and 50 seemed like a good halfway point to stop at. I saved the components into a list so that I'd be able to project the testing batches into PC space later on.
#### k-nearest neighbours
```py
def k_nearest_neighbours(X_train, y_train, X_test, k = 3):
    """
    Predict labels for points in the testing dataset.
    Args:
        X_train (np.ndarray): Points in the training dataset.
        y_train (np.ndarray): Labels in the training dataset.
        X_test (np.ndarray): Points in the testing dataset.
        k (int): The number of nearest neighbours whose label to consider.

    Returns:
        (np.ndarray): List of predicted labels.
    """
    predictions = []

    for test_point in X_test:
        # Compute distances to all training points.
        distances = [np.linalg.norm(train_point - test_point) for train_point in X_train]

        # Determine which points are neighbours.
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        # Determine the most common label amongst neighbours.
        unique_labels, counts = np.unique(k_labels, return_counts = True)
        # In the event of tiebreaks, reduce k. At k = 1 a winner is guaranteed.
        while np.count_nonzero(counts == np.max(counts)) != 1 :
            k -= 1
            unique_labels, counts = np.unique(y_train[np.argsort(distances)[:k]], return_counts = True)
        
        predictions.append(unique_labels[np.argmax(counts)])
    
    return np.array(predictions)
```

I implemented k-NN only with the help of `numpy` operations. During this run I experimented with various hyperparameters because the k-NN predictions were worse than expected, but left it at 10 dimensions and a k of 5 for the visualisation. I shall discuss the results of all visualisations from the run below. 
### Runs across hyperparameters
As the steps of a single run of analysis have been implemented and tested, I prepared a testing regime to perform multiple runs, over a range of hyperparameters. In this section I describe the process of choosing the range of hyperparameters and metrics of evaluation.
#### Testing regime
```py
def test_suite(hp_n, hp_d, hp_k) :
    predictions = {}  # Dictionary to store predictions for each (d, k) pair

    for n in range(hp_n) :
        print('Run #' + str(n + 1) + ', d = ')
        training_dataset, holdout_dataset = holdout(batch_dataset, list(set(authors)), n)
        training_dataset, holdout_dataset = common_tokens(training_dataset, holdout_dataset)
        training_dataset, holdout_dataset = centre(training_dataset, holdout_dataset)
        training_dataset, holdout_dataset = standardise(training_dataset, holdout_dataset)
        components, explained_variance = pca_power_iteration(training_dataset, hp_d, 2000, False)
        
        for d in tqdm(range(1, hp_d + 1)) :
            training_batches = np.array([transform_to_pc(training_dataset[batch], components[:d]) for batch in training_dataset])
            training_labels = np.array([batch.split('_')[1] for batch in training_dataset])
            testing_batches = np.array([transform_to_pc(holdout_dataset[batch], components[:d]) for batch in holdout_dataset])
            
            for k in range(1, hp_k + 1) :
                if n == 0 :
                    predictions[(d, k)] = k_nearest_neighbours(training_batches, training_labels, testing_batches, k)
                else :
                    predictions[(d, k)] = np.append(predictions[(d, k)], k_nearest_neighbours(training_batches, training_labels, testing_batches, k))
                    
    return predictions
```

The regime performs runs for every possible combination of `n` (seed), `d` (dimensions/PCs), and `k` (the k in k-NN), up to `hp_n`, `hp_d`, and `hp_k` respectively. For every `n`, different batches are separated as holdout. This influences the results of feature selection, normalisation and PCA. Then, for every `d`, the batches are projected onto the first `d` components. `d` influences in how many dimensions k-NN will use to calculate distances between batches later on. `d` is looped inside of `n`, because PCA, done once, has already calculated all the PCs that any `d` will require. Lastly, for every `k`, the holdout batches are checked against `k` of their nearest neighbours. `k` is looped inside of `d`, because any `k` will share the same list of training and holdout batches. The regime aggregates all the predictions in a 2-tuple dictionary for each `d` and `k`. `n` is not tracked, as the seed is not an optimisable hyperparameter.

`hp_d` influenced the largest amount of PCs to find distances with. Since later PCs explain less of the variance, there should come a point where the added accuracy from the next PC is not worth the complexity of calculating distances in an additional dimension. The optimal `d` should be between 1 and `hp_d`, so `hp_d` had to be a higher value than was reasonable.

![](images/variance.png)

One popular criterion is to find the *elbow* on a Scree plot, where the PCs right of the elbow no longer lose as much variance as is expected. These PCs may encode noise instead of any useful findings, and can be dropped. However, during this run the computed PCs were too well-distributed to find such an elbow. I even used a logarithmic plot to accentuate the differences between PCs at lower values, but past PC10 a linear trend is observed, meaning that the drop in explained variance follows a smooth logarithmic curve. Another is the *Kaiser criterion*: to drop PCs that don't explain at least 1 unit of variance. Normalised dimensions contribute 1 unit of variance each, and in this view, to include PCs that don't explain as much as an untransformed dimension defeats the purpose of PCA. This point is at PC35, but only 68% of all variance is explained by the first 35 PCs, meaning that a third of potentially meaningful variance could be lost. Finally, I can find some cumulative variance threshold, that can serve as a good enough approximation of the original dimension space. Popular figures include 90% to 97% (3 standard deviations). However, being so well distributed, the first 50 PCs in this run only explain 77% of all variance. Calculating 50 PCs to a tolerance of `1e-8` or 2000 steps, whichever came first, took my workstation over 3 minutes, and I'd have to redo this `hp_n` times. I was balancing between the accuracy of the data and a feasible computation time, and decided that 50 would be an acceptable range for `d`.

`hp_k` influenced the maximum number of neighbours whose labels to consider during predictions. Once again, the optimal `k` should be within the range between 1 and `hp_k`, so `hp_k` had to be a higher value than reasonable.

```
197 batches of human with an average of 199.57 texts per batch.
36 batches of vicuna-13b with an average of 199.81 texts per batch.
33 batches of koala-13b with an average of 200.45 texts per batch.
29 batches of oasst-pythia-12b with an average of 199.10 texts per batch.
28 batches of gpt-3.5-turbo with an average of 199.04 texts per batch.
27 batches of alpaca-13b with an average of 199.96 texts per batch.
25 batches of gpt-4 with an average of 197.80 texts per batch.
22 batches of RWKV-4-Raven-14B with an average of 199.64 texts per batch.
22 batches of claude-v1 with an average of 204.55 texts per batch.
19 batches of chatglm-6b with an average of 199.53 texts per batch.
18 batches of mpt-7b-chat with an average of 195.33 texts per batch.
18 batches of vicuna-7b with an average of 198.78 texts per batch.
18 batches of palm-2 with an average of 195.94 texts per batch.
18 batches of fastchat-t5-3b with an average of 205.50 texts per batch.
17 batches of dolly-v2-12b with an average of 194.53 texts per batch.
16 batches of stablelm-tuned-alpha-7b with an average of 203.06 texts per batch.
16 batches of claude-instant-v1 with an average of 194.81 texts per batch.
12 batches of llama-13b with an average of 199.42 texts per batch.
7 batches of wizardlm-13b with an average of 198.71 texts per batch.
7 batches of gpt4all-13b-snoozy with an average of 203.57 texts per batch.
6 batches of guanaco-33b with an average of 208.00 texts per batch.
```

Some samples (batches) are underrepresented, which may skew the results of k-NN. Theoretically, this should be true for a high `k` and with a lack of batches. For example, there are only 6 batches of *guanaco-33b*. In any run, the training dataset has 5 batches and the holdout dataset has 1. When predicting the label for this holdout batch at `k` = 15, even if all of the training batches of *guanaco-33b* are in the neighbourhood, there will still be 10 other labels, guaranteeing that *guanaco-33b* is not the label of majority. As the largest plurality the correct label may still be chosen, but if the *guanaco-33b* cluster is around another cluster that has a larger population, that cluster's label will be predicted instead. At `k` = 15, 3 out of the 21 labels will be guaranteed to never be a majority of any neighbourhood (and will thus affect precision negatively), so I considered this an acceptable maximum value for `hp_k`.

Since the holdout would only contain one batch from each author, every run for each `d` and `k` would only have one prediction per label. Associated metrics like the confusion matrix and F-scores cannot be reliably interpreted with one-shot sample sizes, so I used Monte-Carlo cross-validation to separate the holdout in different ways for each `d` and `k` as described earlier. `hp_n` influenced the number of runs, and I chose 30, following the popular rule of thumb of the acceptable sample size.
#### Confusion matrix and F-score


## Results
### Visualising a single evaluation
![](images/PCA.png)

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