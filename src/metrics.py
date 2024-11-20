""" Each metric takes in a list of gold and predicted labels, and a k value to truncate the predicted ranking."""


def rp_at_k(gold: list, predicted: list, k: int):
    """
    Calculate Rank Precision at K (RP@K)

    Parameters:
    - gold: List containing the true relevant items
    - predicted: List containing the predicted items in ranked order
    - k: Top K items to consider

    Returns:
    - RP@K (Rank Precision at K) value
    """

    # Ensure k is not greater than the length of the gold list
    gold_k = min(k, len(gold))

    # Retrieve the top K predicted items
    top_k_predicted = predicted[:k]

    # Count the number of true positives in the top K
    true_positives = sum(1 for item in top_k_predicted if item in gold)

    # Calculate RP@K
    rp_at_k = true_positives / gold_k if gold_k > 0 else 0.0

    return rp_at_k


def recall_at_k(gold: list, predicted: list, k: int):
    """
    Calculate Recall at K (Recall@K)

    Parameters:
    - gold: List containing the true relevant items
    - predicted: List containing the predicted items in ranked order
    - k: Top K items to consider

    Returns:
    - Recall@K) (Recall at K) value
    """

    rank = [x in gold for x in predicted]
    recall = sum(rank[:k]) / len(gold)
    return recall


def entity_infer_custom_metric_at_k(gold: list, predicted: list, k: int):
    """
    Custom metric to check:
    1. If the first element of the split ' > ' texts matches (index 0).
    2. If the overall recall at K is met (using recall_at_k).

    Parameters:
    - gold: List containing the true relevant items (assumed to contain ' > ' separated strings)
    - predicted: List containing the predicted items in ranked order (also ' > ' separated strings)
    - k: Top K items to consider

    Returns:
    - A combined score that considers the first element and recall_at_k.
    """

    # Ensure k is not greater than the length of the gold list
    k = min(k, len(predicted))

    # Extract the first part (0th index) from ' > ' separated texts in gold and predicted
    gold_first_part = [item.split(' > ')[0] for item in gold]
    predicted_first_part = [item.split(' > ')[0] for item in predicted[:k]]

    # Check if the first parts match between gold and predicted
    first_part_matches = sum(1 for item in predicted_first_part if item in gold_first_part)

    # Calculate recall_at_k for the overall match
    recall = recall_at_k(gold, predicted, k)

    # Combine both metrics: the ratio of first part matches and recall
    combined_score = (first_part_matches / k) * 0.5 + recall * 0.5

    return combined_score