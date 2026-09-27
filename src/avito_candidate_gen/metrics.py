""" 
Метрика оценки качества кадрогенерации.
Для каждого запроса считается доля релевантных объявлений, 
попавших в предсказанный топ-50
Затем значение усредняется по всем запросам.
"""

def recall_at_50(
    predictions: dict[str, list[str]], relevant: dict[str, set[str]]
) -> float:
    scores = []

    for query_id, rel_items in relevant.items():
        top50 = set(predictions.get(query_id, [])[:50])
        hit = len(top50 & rel_items)
        scores.append(hit / len(rel_items))

    return sum(scores) / len(scores)
