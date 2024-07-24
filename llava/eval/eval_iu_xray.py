# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import os
import re

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

NUM_SECONDS_TO_SLEEP = 0.5


def compute_NLG_scores(nlg_metrics: list[str], gen_sents_or_reports: list[str], ref_sents_or_reports: list[str]) -> dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: list[str]):
        """
        The compute_score methods of the scorer objects require the input not to be list[str],
        but of the form:
        generated_reports =
        {
            "image_id_0" = ["1st generated report"],
            "image_id_1" = ["2nd generated report"],
            ...
        }

        Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
        following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) implementation (https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py):
        see lines 132 and 133
        """
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " .").replace('\n',''))]

        return sents_or_reports_converted
    """
    Computes NLG metrics that are specified in metrics list (1st input argument):
        - Bleu 1-4
        - Meteor
        - Rouge-L
        - Cider-D

    Returns a dict that maps from the metrics specified to the corresponding scores.
    """
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    #if "meteor" in nlg_metrics:
    #    scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    return nlg_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IU X-ray evaluation.')
    parser.add_argument('-a', '--answer')
    parser.add_argument('-g', '--ground-truth')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    f_ans = open(os.path.expanduser(args.answer))
    f_gt = open(os.path.expanduser(args.ground_truth))

    result_file = open(f'{args.output}', 'w')

    ans_text_list = []
    gt_test_list = []
    idx = 0
    for ans_js, gt_js in zip(f_ans, f_gt):
        ans = json.loads(ans_js)
        gt_ans = json.loads(gt_js)
        print('ans: ',ans)
        print('get ans: ',gt_ans)
        if gt_ans['text']:
            ans_text_list.append(ans['text'])
            gt_test_list.append(gt_ans['text'])
        '''
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        '''
        idx += 1
        print(idx)
    #
    METRIC_NAMES = ["bleu","meteor","rouge"]
    score_metric = compute_NLG_scores(METRIC_NAMES,ans_text_list,gt_test_list)
    print('Score Metric:')
    print(score_metric)
    result_file.write(json.dumps(score_metric) + '\n')
    result_file.flush()
    result_file.close()
