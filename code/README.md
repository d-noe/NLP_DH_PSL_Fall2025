# Code

This folder hosts the notebooks and code (in Pyhton) used in the different tutorials and hands-on sessions. The proposed [set-ups](#setup), and [contents](#content) of the sessions are described below.


- [Week 1](#code_week1): Familiarization with BERT-like models, using `transformers` package. Generation of embedding vectors and visualization; applications on word sense disambiguation, and semantic shifts exploration.
- [Week 2](#code_week2): Topic Modeling: follow a step-by-step implementation of a (simplified) version of BERTopic relying on `sentence_transformers` model representations and compare the output of different topic models. Experiments illustrated with a corpus of 19th century American recipes, and UN General Debate speeches.
- [Week 3](#code_week3): Supervised Learning: Tutorial of BERT-like model fine-tuning applied to book genre prediction (compared with document representation-based baselines). Hands-on applied to "literary canon" prediction: design your own classifier and reflect on fairness issues in ML.
- [Week 4](#code_week4): Generative LLMs interactions: Tutorial on how to interact with LLMs (via diverse APIs), and hands-on session on devising a questionnaire to assess LLMs behaviors.

## Setups <a name="setup"></a>

Feel free to use the notebooks, either locally, or using hosted services such as Jupyter Binder and Google Colab.

### Running on your machine

You can use the `requirements.txt` file provided at the root of this repository. In your virtual environment, `cd` to repo root, and run:

```bash
pip install -r requirements.txt
```

### Binder

You can launch the projects on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/d-noe/NLP_DH_PSL_Fall2025/HEAD)

> [!WARNING]
> It can take some time to build the image on Binder.

Binder can be handy to have the repository in a jupyter-lab hosted environment. However, it does not provide extensive memory nor computational resources. Thus, it is not fitted to manipulate large data or to use pre-trained language models with large number of parameters.

### Colab

The notebooks are provided in Google Colab. It provides a convenient way to run the experiments and offers computational resources that should be sufficient for the content of this course in the free-tier (including GPU and TPU runtime access).



## Content <a name="content"></a>

### Week 1 — 29.10 <a name="code_week1"></a>

- [Discover_BERT.ipynb](./1_bert_training/Discover_BERT.ipynb): Familiarize with BERT-like models. Overview of the architecture. Visulisation of attention mechanism. 
- [Tutorial_1_WSD.ipynb](./1_bert_training/Tutorial_1_WSD.ipynb): Familiarize with BERT-like models. Generation of embedding vectors and visualisation. Exemplified with Word Sense Disambiguation application.
- [Hands-on_1_SS.ipynb](./1_bert_training/Hands_on_1_SS.ipynb): Reproduce and expand tutorial's content. Explore Semantic Shifts from LM's lense based on historical newspaper data from [Living With Machines](https://livingwithmachines.ac.uk) initiative.

**Main libraries**: `transformers`, `bertviz`, (`scikit-learn`, `pandas`, `altair`)

<a name="code_supp_1"></a>
<details><summary>To go further</summary> 

- Implement the *attention mechanism* from the [*Attention is All You Need* (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) cornerstone paper in a [Colab Notebook](https://colab.research.google.com/drive/1tm0_Usqkavr0h1Jk0f-ukcykI78xmcfW#scrollTo=tSWEk4ttUgQH) by Alexander "Sasha" Rush. Or read their detailed walkthrough post: [*The Annotated Transformer*](http://nlp.seas.harvard.edu/annotated-transformer/).

</details>

### Week 2 — 05.11 <a name="code_week2"></a>

- [Tutorial_2_MyBERTopic.ipynb](./2_topic_modeling/Tutorial_2_MyBERTopic.ipynb): Implement your own (simplified) version of BERTopic and explore a corpus of 19th century recipes.
- [Hands_on_2_CompareTM.ipynb](./2_topic_modeling/Hands_on_2_CompareTM.ipynb): Apply different topic modeling algorithms on a corpus of UN General Debate speeches transcripts from 1946 until today. Explore time and space, and try to find the best methods to decipher what is discussed during these assemblies!

**Main libraries**: `sentence_transformers`, `BERTopic`, `gensim`, `pyLDAvis`, `sklearn`, `umap`, `hdbscan`

<a name="code_supp_2"></a>
<details><summary>To go further</summary> 

- [Tutorial - Topic Modeling with BERTopic](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-#scrollTo=AXHLDxJdRzBi): A tutorial and overview of the different functionalities of `BERTopic` (Author unknown?).
- [Tutorial - LDA Topic Modeling with `sklearn` and visualization with `pyLDAvis`](https://nbviewer.org/github/bmabey/pyLDAvis/blob/master/notebooks/LDA%20model.ipynb).
- [Understanding and Using Common Similarity Measures for Text Analysis](https://doi.org/10.46430/phen0089): A detailed tutorial on computing distances on text document (using BoW-like representations) in Python, applied to data from the [EarlyPrint](https://earlyprint.org) initiative. ©John R. Ladd (2020).

</details>

### Week 3 — 12.11 <a name="code_week3"></a>

- [Tutorial_3_SFT.ipynb](./3_supervised/Tutorial_3_SFT.ipynb): Fine-tune a BERT-like model for literary genre classification based on 5-sentences long book chunks. Compare the results with classification performed by standard classifiers trained on document representations (BoW, TF-IDF, SentenceTransformers' embeddings).
- [Hands_on_3_CanonChallenge.ipynb](./3_supervised/Hands_on_3_CanonChallenge.ipynb): Your time to devise a classifier for "canonicity" prediction based on 5-sentences long excerpts of French-language novels. Reflect about the data, the models, and the fairness implications of both. Implement you classifier and submit your predictions to the [*Performance & Fairness class shared task*](https://leaderboard-performance-fairness.streamlit.app): [https://tinyurl.com/canon-pf](https://tinyurl.com/canon-pf)!

**Main libraries**: `transformers`, `pytorch`, `sklearn`

<a name="code_supp_3"></a>
<details><summary>To go further</summary> 

- [Tutorial: Fine-tuning ](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_code_llm_on_single_gpu.ipynb): *Fine-tuning a Code LLM on Custom Code on a single GPU*, by Maria Khalusova.
- Tutorial: Interpreting BERT's classification decisions: *Interpreting the Prediction of BERT Model for Text Classification*, by Ruben Winastwan. ([Blog post](https://towardsdatascience.com/interpreting-the-prediction-of-bert-model-for-text-classification-5ab09f8ef074/) | [Notebook](https://github.com/mrubenw/medium-resources/blob/main/BERT_Captum/Bert_captum.ipynb))
- [Fairness with the `dalex` Python Package](https://dalex.drwhy.ai/python-dalex-fairness.html).

</details>

### Week 4 — 19.11 <a name="code_week4"></a>

- [Tutorial_4_LLM_Interaction.ipynb](./4_causal/Tutorial_4_LLM_Interaction.ipynb): Learn how to use open-weight LLM via the `transformers` library, run and query LLMs locally with `ollama`, or interact with diverse providers through APIs and requests.
- [Hands_on_4_EvalLLM.ipynb](./4_causal/Hands_on_4_EvalLLM.ipynb): Write a multiple choice questionnaire and apply it to LLMs.

**Main libraries**: `transformers`, `ollama`, `openai`, `requests`

<a name="code_supp_4"></a>
<details><summary>To go further</summary>

**Revisit previous sessions with LLMs!**
- [Week_1](#code_week1): 
    - Re-annotate the data with an LLM and observe potential differences, measure aggreement with humans with κ index
    - Extract features from a generative LLM instead of BERT
    - Improve the OCRed text via prompting LLMs (& find methods to evaluate improvement)
- [Week_2](#code_week2): 
    - Add a LLM-based component to summarize topics or provide more meaningful topic labels
    - Replace the document embedder of BERTopic with features extracted from a LLM
- [Week_3](#code_week3): 
    - Prompt LLMs to do zero-/few- shot classification (try diverse prompts, number of examples, etc.)
        - of book genres
        - or canonicity (upload your predictions on the Shared Task app!)

- [Glimpse at Data Curation for LLM training](https://colab.research.google.com/drive/1EhHV3wZEjCltcm4idXdX1TnmEgdV1QhG?usp=sharing): Colab notebook to explore data curation process: language identification & quality filtering, by Rose E Wang.
- [Interrogating a National Narrative with GPT-2](https://programminghistorian.org/en/lessons/interrogating-national-narrative-gpt): Using Generated Texts to Interrogate the *Brexit* Narrative (Lesson), by Chantal Brousseau.
- [Text Classification using LLMs](https://github.com/skorch-dev/skorch/blob/master/notebooks/LLM_Classifier.ipynb): Using the `skorch` library for zero-shot classification with LLMs.

</details>