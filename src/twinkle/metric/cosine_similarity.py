# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from .base import Metric


class CosineSimilarityMetric(Metric):

    def __call__(self, inputs, outputs, **kwargs):
        from sklearn.metrics.pairwise import (paired_cosine_distances, paired_euclidean_distances,
                                              paired_manhattan_distances)
        from scipy.stats import pearsonr, spearmanr
        embeddings1 = outputs['sentence1']
        embeddings2 = outputs['sentence2']
        labels = inputs['labels']
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        return {
            'pearson_cosine': eval_pearson_cosine,
            'pearson_euclidean': eval_pearson_euclidean,
            'pearson_manhattan': eval_pearson_manhattan,
            'pearson_dot_product': eval_pearson_dot,
            'spearman_cosine': eval_spearman_cosine,
            'spearman_euclidean': eval_spearman_euclidean,
            'spearman_manhattan': eval_spearman_manhattan,
            'spearman_dot_product': eval_spearman_dot,
        }