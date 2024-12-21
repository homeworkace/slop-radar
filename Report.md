# slop-radar: An attempt in identifying human and AI-generated texts

## Abstract
*This is a single paragraph that summarises the aim and findings of your project.*
- summarise every other section
## Introduction
*This section places your project in a machine learning context as exemplified by the academic literature. You can amplify the aim as stated in the abstract and explain why this aim is interesting and relevant. You might introduce your dataset(s) and explain any particular problems both with the dataset and with known investigations of this dataset.*

When OpenAI released ChatGPT in \*insert date\*, it disrupted multiple industries and many professionals considered the ways they could use it to improve their productivity. Unfortunately,  instead of viewing it as just another tool, many users have taken this to mean replacing entire tasks, like writing emails or drafting documents from scratch. Our current state-of-the-art *Large Language Models* suffer from a high rate of mistakes and an inability to fact check. People find their messages distant and impersonal as they lack the context that a human would have. LLMs have seen widespread use in an academic setting, and have had an equally widespread pushback as a form of academic dishonesty. Traditional plagiarism-checking services like Turnitin have even started to offer detection of AI-written work. (cite) Their ability to write in a convincing and authoritative manner with minimal effort has greatly simplified the creation of disinformation campaigns and scams. (cite)

Just as literacy has become necessary for all after the industrial age, I believe that digital literacy is necessary for all to stay safe in the near future. I consider myself an early adopter as I started to [download and run such models](https://huggingface.co/cognitivecomputations/WizardLM-33B-V1.0-Uncensored) off my own computer in July of 2023. [Enthusiasts in the LLM space](https://www.reddit.com/r/LocalLLaMA/comments/17xwuno/what_do_you_think_about_gptisms_polluting/) who have seen enough generated text know their shortcomings more intimately and have developed a better sense of whether any text is written by a human. They try to identify what they call *GPT-isms* or *slop*: choice phrases and figures of speech that are stereotypical of ChatGPT or other LLMs, which can be useful for a layperson to detect AI-generated text themselves.

(Literature review? They focus on the effectiveness of detection, but they do not seem to draw conclusions that the general public can follow, and it may be impractical to rely on detection algorithms at all times.)

In this project, I prepare and analyse a corpus of texts written by a roster of small LLMs, and aim to identify choice vocabulary that distinguishes between human and LLM-generated texts. I also perform further exploratory analysis in an attempt to identify writing styles between models. Finally, I evaluate the effectiveness of my methodology as compared to that of existing studies, and conclude with some actionable steps to identify AI-generated text.
## Background
*Here you should explain in your own words how your algorithms work. This is where you can demonstrate that you understand the tools and models that you used. Marks will be gained accordingly.*
### Dataset
(talk about LMArena)
### Algorithms
(list algorithms used in preparing the dataset and in analysis)
## Methodology
*Set out how you explored the dataset and/or algorithm modifications. For example, you might have decided to use cross-validation; if so, explain why this technique was necessary.*

Preparing the dataset
- Extracting text-author pairs
- Vectorising and grouping into batches

A single evaluation
- Separating holdout
- Feature selection
- Centring and standardising
- PCA with power iteration

Testing run
- 10 evaluations
- other evaluations
- k-Nearest Neighbours for F-score/confusion matrix
## Results
*Your results must be stated clearly. Tables are recommended. You should cross-reference to the experiments that you described in the Methodology section.*

various observations from single evaluation
accuracy of k-nn
## Evaluation
*This is a chance to demonstrate a critical awareness of the strengths and weaknesses of your project. Remember that a research project is judged by referring to the stated aim. A project does not have to succeed! Marks are allocated for how you undertook the project and for understanding and insight. A very ambitious aim might be unachievable within the terms of this project.*

conclusions from single evaluation (to tell apart human from synthetic writing)
conclusions from testing run (to tell apart writing styles)
## Conclusions
*State succinctly your findings and how they relate to your aim.*
bruh
## References
*List any academic work (such as a book or a research paper) that you refer to in the main body of the report.*
bruh