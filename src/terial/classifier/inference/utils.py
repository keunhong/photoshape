

def compute_weighted_scores(inference_dict, mat_by_id, sort=True,
                            force_substances=False):
    for seg_id, seg_topk in inference_dict['segments'].items():
        if 'substance' not in seg_topk:
            continue
        compute_weighted_scores_single(seg_topk, mat_by_id,
                                       sort, force_substances)

    return inference_dict


def compute_weighted_scores_single(seg_topk, mat_by_id, sort=True,
                                   force_substances=False,
                                   weight_substances=False):
    total_score = 0
    for match in seg_topk['material']:

        pred_subst  = match.get('pred_substance',
                                mat_by_id[int(match['id'])].substance)

        if force_substances:
            top_subst = match.get('minc_substance',
                                  seg_topk['substance'][0]['name'])
            subst_score = 1.0 if (top_subst == pred_subst) else 0.0
        elif weight_substances:
            subst_scores = {
                m['name']: m['score'] for m in seg_topk['substance']
            }
            subst_score = subst_scores[pred_subst]
        else:
            subst_score = 1.0
        score = match['score'] * subst_score
        match['weighted_score'] = score
        total_score += score
    if sort:
        seg_topk['material'].sort(key=lambda match: -match['weighted_score'])
        seg_topk['material'] = seg_topk['material'][:10]

    if total_score > 0 and force_substances:
        for match in seg_topk['material']:
            match['weighted_score'] /= total_score

    return seg_topk



