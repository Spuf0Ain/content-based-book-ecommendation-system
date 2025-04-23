from typing import List, Set, Any

def precision_at_k(recommended_items: List[Any], relevant_items: Set[Any], k: int) -> float:
    """
    Calculates Precision@k: the proportion of recommended items in the top k
    that are actually relevant.

    Args:
        recommended_items (List[Any]): A list of recommended item IDs, ordered by relevance.
        relevant_items (Set[Any]): A set of item IDs that are considered relevant
                                   (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: The Precision@k score (between 0.0 and 1.0).
    """
    if k <= 0:
        return 0.0
    if not recommended_items or not relevant_items:
        return 0.0

    # Take the top k recommendations
    top_k_recommendations = recommended_items[:k]

    # Count how many of the top k recommendations are in the relevant set
    hits = sum(1 for item in top_k_recommendations if item in relevant_items)

    # Precision = (Number of relevant items recommended@k) / k
    precision = hits / k
    return precision

def recall_at_k(recommended_items: List[Any], relevant_items: Set[Any], k: int) -> float:
    """
    Calculates Recall@k: the proportion of all relevant items that are
    successfully recommended in the top k.

    Args:
        recommended_items (List[Any]): A list of recommended item IDs, ordered by relevance.
        relevant_items (Set[Any]): A set of item IDs that are considered relevant
                                   (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: The Recall@k score (between 0.0 and 1.0).
    """
    if not relevant_items:
        return 0.0
    if k <= 0:
        return 0.0
    if not recommended_items:
        return 0.0

    # Take the top k recommendations
    top_k_recommendations = recommended_items[:k]

    # Count how many of the top k recommendations are in the relevant set
    hits = sum(1 for item in top_k_recommendations if item in relevant_items)

    # Recall = (Number of relevant items recommended@k) / (Total number of relevant items)
    recall = hits / len(relevant_items)
    return recall

def f1_score_at_k(recommended_items: List[Any], relevant_items: Set[Any], k: int) -> float:
    """
    Calculates the F1-Score@k, the harmonic mean of Precision@k and Recall@k.

    Args:
        recommended_items (List[Any]): A list of recommended item IDs, ordered by relevance.
        relevant_items (Set[Any]): A set of item IDs that are considered relevant
                                   (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: The F1-Score@k (between 0.0 and 1.0).
    """
    prec = precision_at_k(recommended_items, relevant_items, k)
    rec = recall_at_k(recommended_items, relevant_items, k)

    # Harmonic mean: 2 * (Precision * Recall) / (Precision + Recall)
    if prec + rec == 0:
        return 0.0 
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
        return f1

if __name__ == '__main__':
    # Assume 'Book A' to 'Book J' are possible items
    all_items = {f'Book {chr(ord("A") + i)}' for i in range(10)}

    # Example 1: Good recommendations
    recommendations1 = ['Book B', 'Book D', 'Book A', 'Book F', 'Book H'] # Top 5 recs
    relevant_set1 = {'Book A', 'Book B', 'Book C', 'Book D', 'Book E'}    # Ground truth

    k = 5
    precision1 = precision_at_k(recommendations1, relevant_set1, k)
    recall1 = recall_at_k(recommendations1, relevant_set1, k)
    f1_score1 = f1_score_at_k(recommendations1, relevant_set1, k)

    print(f"Example 1 (k={k}):")
    print(f"  Recommendations: {recommendations1}")
    print(f"  Relevant Items: {relevant_set1}")
    print(f"  Precision@{k}: {precision1:.4f}") # Hits: B, D, A (3) -> 3/5 = 0.6
    print(f"  Recall@{k}: {recall1:.4f}")      # Hits: B, D, A (3) / Total Relevant (5) -> 3/5 = 0.6
    print(f"  F1-Score@{k}: {f1_score1:.4f}")    # 2 * (0.6 * 0.6) / (0.6 + 0.6) = 0.72 / 1.2 = 0.6

    # Example 2: Poor recommendations
    recommendations2 = ['Book Z', 'Book Y', 'Book X', 'Book W', 'Book V'] # Top 5 recs
    relevant_set2 = {'Book A', 'Book B', 'Book C', 'Book D', 'Book E'}    # Ground truth

    k = 5
    precision2 = precision_at_k(recommendations2, relevant_set2, k)
    recall2 = recall_at_k(recommendations2, relevant_set2, k)
    f1_score2 = f1_score_at_k(recommendations2, relevant_set2, k)

    print(f"\nExample 2 (k={k}):")
    print(f"  Recommendations: {recommendations2}")
    print(f"  Relevant Items: {relevant_set2}")
    print(f"  Precision@{k}: {precision2:.4f}") # Hits: 0 -> 0/5 = 0.0
    print(f"  Recall@{k}: {recall2:.4f}")      # Hits: 0 / Total Relevant (5) -> 0/5 = 0.0
    print(f"  F1-Score@{k}: {f1_score2:.4f}")    # 0.0

    # Example 3: Different K
    k = 3
    precision3 = precision_at_k(recommendations1, relevant_set1, k) # Check top 3: B, D, A
    recall3 = recall_at_k(recommendations1, relevant_set1, k)      # Check top 3: B, D, A
    f1_score3 = f1_score_at_k(recommendations1, relevant_set1, k)

    print(f"\nExample 3 (k={k}, using recommendations1):")
    print(f"  Precision@{k}: {precision3:.4f}") # Hits: B, D, A (3) -> 3/3 = 1.0
    print(f"  Recall@{k}: {recall3:.4f}")      # Hits: B, D, A (3) / Total Relevant (5) -> 3/5 = 0.6
    print(f"  F1-Score@{k}: {f1_score3:.4f}")    # 2 * (1.0 * 0.6) / (1.0 + 0.6) = 1.2 / 1.6 = 0.75

    print("\nMetrics module executed successfully (for testing).")