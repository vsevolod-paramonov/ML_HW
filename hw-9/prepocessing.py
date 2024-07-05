from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import xml.etree.ElementTree as ET
import io



@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """

    with open(filename, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    
    file_content = io.StringIO(file_content.replace('&', '&amp;'))

    file = ET.parse(file_content)
    root = file.getroot()

    try_separation = lambda x: x.split() if x is not None else None
    another_separate = lambda x: tuple(map(int, x.split('-'))) if x is not None else None

    sent_pairs, alignments = [], []


    for child in root:

        eng, czh = child[0].text.split(), child[1].text.split()

        sure, possib = try_separation(child[2].text), try_separation(child[3].text)
        sure, possib = list(map(another_separate, sure)) if sure is not None else [], list(map(another_separate, possib)) if possib is not None else []

        pairs = SentencePair(eng, czh)
        align = LabeledAlignment(sure, possib)

        sent_pairs.append(pairs)
        alignments.append(align)


    return (sent_pairs, alignments)


    

pass


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    
    words_source = np.array([word for sentns in [i.source for i in sentence_pairs] for word in sentns])
    words_target = np.array([word for sentns in [i.target for i in sentence_pairs] for word in sentns])

    words_source_unique, order = np.unique(words_source, return_index=True)
    words_source_unique = words_source[np.sort(order)]

    words_target_unique, order = np.unique(words_target, return_index=True)
    words_target_unique = words_target[np.sort(order)]

    source_dict = {word: token for word, token in zip(words_source_unique, np.arange(len(words_source)))}
    target_dict = {word: token for word, token in zip(words_target_unique, np.arange(len(words_target)))}

    if freq_cutoff is not None:

        unique, counts, order = np.unique(words_source, return_counts=True, return_index=True)
        freq_word_source = words_source[np.sort(order)][counts >=  freq_cutoff]

        unique, counts, order = np.unique(words_target, return_counts=True, return_index=True)
        freq_word_target = words_target[np.sort(order)][counts >=  freq_cutoff]

        source_dict = {word: source_dict[word] for word in freq_word_source}
        target_dict = {word: target_dict[word] for word in freq_word_target}


    return (source_dict, target_dict)



def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    
    checking = lambda x, dictt: all([i in dictt.keys() for i in x])
    check = [all([checking(sentence_pairs[i].source, source_dict), checking(sentence_pairs[i].target, target_dict)]) for i in range(len(sentence_pairs))]
    
    idx = np.arange(len(sentence_pairs))[check]

    outp = []

    for i in idx:

        sent, trgt = sentence_pairs[i].source, sentence_pairs[i].target

        sent, trgt = [source_dict[i] for i in sent], [target_dict[i] for i in trgt]

        outp.append(TokenizedSentencePair(np.array(sent), np.array(trgt)))


    return outp
