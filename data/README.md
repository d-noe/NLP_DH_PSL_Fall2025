# Data 

This folder hosts the data used in the different tutorials and hands-on sessions. Short descriptions are provided below, please refer to the original sources for further information.

- [Week 1](#data_week1)
  - Word Sense Disambiguation: [CoarseWSD-20](#wsd)
  - Semantic Shifts Analysis: [Living With Machines](#lwm)
- [Week 2](#data_week2)
- [Week 3](#data_week3)
- [Week 4](#data_week4)

The [`preprocessing`](./preprocessing/) folder contains pre-processing scripts used to obtain the presented data files. It is included for transparency and to allow reproduction, but was not developed for pedagogical purposes (-> the code is likely quick and dirty).

## Week 1 — 29.10 <a name="data_week1"></a>

### Tutorial 1: Word Sense Disambiguation <a name="wsd"></a>

- File(s): [`word_sense/CoarseWSD-20.csv`](./word_sense/CoarseWSD-20.csv)
- Description:
  - Dataset based on Wikipedia articles excerpts, developed to evaluate and train word disambiguation models based on a selection of 20 expert-selected words (apple, club, bow, bank, crane, square, chair, java, bass, seal, pitcher, arm, mole, pound, deck, trunk, spring, hood, yard, digit).
  - columns: `text`, `target`, `label`, `label_sense`, `split`
- Source: *Daniel Loureiro, Kiamehr Rezaee, Mohammad Taher Pilehvar, & Jose Camacho-Collados. (2021). Analysis and Evaluation of Language Models for Word Sense Disambiguation.* [![arXiv](https://img.shields.io/badge/arXiv-2008.11608-b31b1b.svg)](https://arxiv.org/abs/2008.11608)

### Hands-on 1: Semantic Shifts <a name="lwm"></a>

Data from the [Living With Machines](https://livingwithmachines.ac.uk) programme. Please check their website if interested in *digital history*, and to discover their research on rethinking the impact of technology on people's lives during the Industrial Revolution.

- File(s): [`word_sense/lwm_annot.csv`](./word_sense/lwm_annot.csv)
- Description:
  - Excerpts of british historical newspaper articles (1818-1920), OCR texts, annotated for word sense for four target words (trolley, car, bike, coach).
  - columns: `text`, `annot`, `newspaper_date`, `newspaper_title`, `newspaper_place`, `image_url`, `image_url_conflict_values`, `target`, `year`.
- Source: Ridge, M., Pedrazzini, N., Vieira, J. M. M., Ciula, A., & McGillivray, B. (2024). Language of mechanisation crowdsourcing datasets from the living with machines project. Journal of Open Humanities Data, 10, 33. [Link](https://kclpure.kcl.ac.uk/portal/en/publications/language-of-mechanisation-crowdsourcing-datasets-from-the-living-) ; [Data Repository](https://bl.iro.bl.uk/concern/datasets/1a677294-cbd3-4bc0-b714-d3bbfd2a6da1)

## Week 2 — 05.11 <a name="data_week2"></a>

TBD

## Week 3 — 12.11 <a name="data_week3"></a>

TBD

## Week 4 — 19.11 <a name="data_week4"></a>

TBD