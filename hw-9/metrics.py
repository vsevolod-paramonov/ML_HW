from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """

    intersection = total_predicted = 0
    
    for i in range(len(reference)):

        s_1, s_2, s_3 = set(reference[i].possible), set(predicted[i]), set(reference[i].possible + reference[i].sure)

        intersection += len(s_3 & s_2)

        total_predicted += len(s_2)

    return intersection, total_predicted



def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """

    numerator = denominator = 0
    
    for i in range(len(reference)):

        s_1, s_2 = set(reference[i].sure), set(predicted[i])

        s = len(s_1 & s_2)
        power = len(s_1)

        numerator += s
        denominator += power

    return numerator, denominator


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    
    intersect_precision, pred_precision = compute_precision(reference, predicted)
    intersect_recall, pred_recall =  compute_recall(reference, predicted)
    
    return 1 - (intersect_recall + intersect_precision) / (pred_precision + pred_recall)
