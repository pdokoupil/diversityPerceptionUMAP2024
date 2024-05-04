import numpy as np
import functools
from sklearn.preprocessing import QuantileTransformer


# Some small enough number
NEG_INF = int(-10e6)

def mask_scores(scores, seen_items_mask):
    # Ensure seen items get lowest score of 0
    # Just multiplying by zero does not work when scores are not normalized to be always positive
    # because masked-out items will not have smallest score (some valid, non-masked ones can be negative)
    # scores = scores * seen_items_mask[user_idx]
    # So instead we do scores = scores * seen_items_mask[user_idx] + NEG_INF * (1 - seen_items_mask[user_idx])
    min_score = scores.min()
    # Here we do not mandate NEG_INF to be strictly smaller
    # because rel_scores may already contain some NEG_INF that was set by predict_with_score
    # called previously -> so we allow <=.
    assert NEG_INF <= min_score, f"min_score ({min_score}) is not smaller than NEG_INF ({NEG_INF})"
    scores = scores * seen_items_mask + NEG_INF * (1 - seen_items_mask)
    return scores


# Selects next candidate with highest score
# Calculate scores as (1 - alpha) * mgain_rel + alpha * mgain_div
# that is used in diversification experiments
def select_next(alpha, mgains, seen_items_mask):
    # We ignore seen items mask here
    assert mgains.ndim == 2 and mgains.shape[0] == 2, f"shape={mgains.shape}"
    scores = (1.0 - alpha) * mgains[0] + alpha * mgains[1]
    assert scores.ndim == 1 and scores.shape[0] == mgains.shape[1], f"shape={scores.shape}"
    scores = mask_scores(scores, seen_items_mask)
    return scores.argmax()
    

# Runs incremental diversification on a relevance based recommendation, where:
#   *rel_scores* is 1D numpy array containing estimated relevances of all items
#   *alpha* diversification strength (higher alpha means higher diversification, see formula in select_next above)
#   *items* is a numpy array with all items (zero-based, so basically np.arange(rating_matrix.shape[1]))
#   *diversify_f* is diversity function that cna be called with arbitrary top-k list
#        and that correspond to objective w.r.t. which the diversification should happen
#        in our experiments, this was either CF-ILD, CB-ILD, or BIN-DIV
#   *rating_row* corresponding to implicit feedback from the user for which we generate the recommendation
#       (basically arbitrary numpy array of shape |items| that contain 0/1 implicit feedback)
#   *filter_out_items* which items should not be recommended
#       (typically we do not want to recommend same items twice for single user)
#   *n_items_subset* how many candidate items should we consider, this is crucial otherwise the procedure is not
#       computationally feasible in real-time user study (esp. because of the BIN-DIV)
#       we used 500 in our experiments (is expected to be divisible by 2).
#       We use "random mixture", this means that n_items_subset / 2 items were those with highest relevance
#       and remaining n_items_subset / 2 were selected randomly from the rest of candidate items 
def diversify(k, rel_scores,
            alpha, items, relevance_f, diversity_f,
            rating_row, filter_out_items, n_items_subset=None):

    assert rel_scores.ndim == 1
    assert rating_row.ndim == 1

    # This is going to be the resulting top-k item
    top_k_list = np.zeros(shape=(k, ), dtype=np.int32)

    # Hold marginal gain for each item, objective pair
    mgains = np.zeros(shape=(2, items.size if n_items_subset is None else n_items_subset), dtype=np.float32)

    # Sort relevances
    # Filter_out_items are already propageted into rel_scores (have lowest score)
    sorted_relevances = np.argsort(-rel_scores, axis=-1)

    # If n_items_subset is specified, we take subset of items
    if n_items_subset is None:
        # Mgain masking will ensure we do not select items in filter_out_items set
        source_items = items
    else:
        assert n_items_subset % 2 == 0, f"When using random mixture we expect n_items_subset ({n_items_subset}) to be divisible by 2"
        # Here we need to ensure that we do not include already seen items among source_items
        # so we have to filter out 'filter_out_items' out of the set

        # We know items from filter_out_items have very low relevances
        # so here we are safe w.r.t. filter_out_movies because those will be at the end of the sorted list
        relevance_half = sorted_relevances[:n_items_subset//2]
        # However, for the random half, we have to ensure we do not sample movies from filter_out_movies because this can lead to issues
        # especially when n_items_subset is small and filter_out_items is large (worst case is that we will sample exactly those items that should have been filtered out)
        random_candidates = np.setdiff1d(sorted_relevances[n_items_subset//2:], filter_out_items)
        random_half = np.random.choice(random_candidates, n_items_subset//2, replace=False)
        source_items = np.concatenate([
            relevance_half, 
            random_half
        ])

    # Default number of quantiles is 1000, however, if n_samples is smaller than n_quantiles, then n_samples is used and warning is raised
    # to get rid of the warning, we calculates quantiles straight away
    n_quantiles = min(1000, mgains.shape[1])

    # Mask-out seen items by multiplying with zero
    # i.e. 1 is unseen
    # 0 is seen
    # Lets first set zeros everywhere
    seen_items_mask = np.zeros(shape=(source_items.size, ), dtype=np.int8)
    # And only put 1 to UNSEEN items in CANDIDATE (source_items) list
    seen_items_mask[rating_row[source_items] <= 0.0] = 1
    
    # Build the recommendation incrementally
    for i in range(k):
        # Relevance and diversity
        for obj_idx, obj_func in enumerate([relevance_f, diversity_f]):
            # Cache f_prev
            f_prev = obj_func(top_k_list[:i])
            
            objective_cdf_train_data = []
            # For every source item, try to add it and calculate its marginal gain
            for j, item in enumerate(source_items):
                top_k_list[i] = item # try extending the list
                objective_cdf_train_data.append(obj_func(top_k_list[:i+1]) - f_prev)
                mgains[obj_idx, j] = objective_cdf_train_data[-1]
                
            # Use cdf_div to normalize marginal gains
            mgains[obj_idx] = QuantileTransformer(n_quantiles=n_quantiles).fit_transform(mgains[obj_idx].reshape(-1, 1)).reshape(mgains[obj_idx].shape)
    
        # Select next best item to be added (incrementally) to the recommendation list
        best_item_idx = select_next(alpha, mgains, seen_items_mask)
        best_item = source_items[best_item_idx]
            
        # Select the best item and append it to the recommendation list            
        top_k_list[i] = best_item
        # Mask out the item so that we do not recommend it again
        seen_items_mask[best_item_idx] = 0

    return top_k_list


# Wrapper for loading pre-trained item-item matrix (attached in our OSF repository)
class EASER_pretrained:
    def __init__(self, all_items, **kwargs):
        self.all_items = all_items

    def load(self, path):
        self.item_item = np.load(path)
        assert self.item_item.shape[0] == self.item_item.shape[1] == self.all_items.size
        return self

    def predict_with_score(self, selected_items, filter_out_items, k):
        candidates = np.setdiff1d(self.all_items, selected_items)
        candidates = np.setdiff1d(candidates, filter_out_items)
        user_vector = np.zeros(shape=(self.all_items.size,), dtype=self.item_item.dtype)
        if selected_items.size == 0:
            return np.zeros_like(user_vector), user_vector, np.random.choice(candidates, size=k, replace=False).tolist()
        user_vector[selected_items] = 1
        probs = np.dot(user_vector, self.item_item)

        # Here the NEG_INF used for masking must be STRICTLY smaller than probs predicted by the algorithms
        # So that the masking works properly
        assert NEG_INF < probs.min()
        # Mask out selected items
        probs[selected_items] = NEG_INF
        # Mask out items to be filtered
        probs[filter_out_items] = NEG_INF
        return probs, user_vector, np.argsort(-probs)[:k].tolist()

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        return self.predict_with_score(selected_items, filter_out_items, k)[2]
    

################################################ NOTE ######################################
## Expected usage:
# algo = EASER_pretrained(items)
# algo = algo.load("item_item.npy")
# rel_scores, user_vector, ease_pred = algo.predict_with_score(elicitation_selected, elicitation_selected, k)
# diversified_top_k = diversify(k, rel_scores, alpha, items, cf_ild, rating_row=user_vector, filter_out_items=elicitation_selected, n_items_subset=500)
#############################################################################################