from argparse import ArgumentParser
import json
from tqdm import tqdm
import sys

sys.path.append("/mnt/cache/xingsen/ego4d/")
from ego4d.evaluation.sta_metrics import STAMeanAveragePrecision, OverallMeanAveragePrecision
import numpy as np


class EvaluationException(Exception):
    pass


class MissingUidException(EvaluationException):
    def __init__(self, uids: np.array) -> None:
        self.uids = uids

    def __str__(self):
        return (
                "The following uids are missing: "
                + ", ".join(self.uids.astype(str))
                + "."
        )


class MissingResultsJSONFieldException(EvaluationException):
    def __init__(self, field: str) -> None:
        self.field = field

    def __str__(self):
        return "Missing '{}' field in results JSON".format(self.property)


class MissingPropertyException(EvaluationException):
    def __init__(self, property: str, uid: int = None) -> None:
        self.property = property
        self.uid = uid

    def __str__(self):
        message = "Missing '{}' property".format(self.property)
        if self.uid is not None:
            message += " for uid {}.".format(self.uid)
        return message


class InvalidPropertyException(EvaluationException):
    def __init__(self, property: str, value, uid: int = None) -> None:
        self.property = property
        self.value = value
        self.uid = uid

    def __str__(self):
        message = "Invalid '{}' value '{}' for property".format(self.value, self.property)
        if self.uid is not None:
            message += " for uid {}.".format(self.uid)
        return message


class UnsupportedChallengeException(EvaluationException):
    def __init__(self, challenge: str) -> None:
        self.challenge = challenge

    def __str__(self):
        return "Challenge '{}' declared in results JSON not supported".format(self.challenge)


class UnsupportedVersionException(EvaluationException):
    def __init__(self, version: str) -> None:
        self.version = version

    def __str__(self):
        return "Version '{}' declared in results JSON not supported".format(self.version)


def validate_results(results, annotations):
    if 'results' not in results:
        raise MissingResultsJSONFieldException('results')

    if 'version' not in results:
        raise MissingResultsJSONFieldException('version')

    if 'challenge' not in results:
        raise MissingResultsJSONFieldException('challenge')

    if results['challenge'] != 'ego4d_short_term_object_interaction_anticipation':
        raise UnsupportedChallengeException(results['challenge'])

    if results['version'] != '1.0':
        raise UnsupportedVersionException(results['version'])

    for k, vv in results['results'].items():
        for v in vv:
            if 'box' not in v:
                raise MissingPropertyException('box', k)
            if not isinstance(v['box'], list) or len(v['box']) != 4:
                raise InvalidPropertyException('box', v['box'], k)
            if 'noun_category_id' not in v:
                raise MissingPropertyException('noun_category_id', k)
            if not isinstance(v['noun_category_id'], int):
                raise InvalidPropertyException('noun_category_id', v['noun_category_id'], k)
            if 'verb_category_id' not in v:
                raise MissingPropertyException('verb_category_id', k)
            if not isinstance(v['verb_category_id'], int):
                raise InvalidPropertyException('verb_category_id', v['verb_category_id'], k)
            if 'time_to_contact' not in v:
                raise MissingPropertyException('time_to_contact', k)
            if not isinstance(v['time_to_contact'], float):
                raise InvalidPropertyException('time_to_contact', v['time_to_contact'], k)
            if 'score' not in v:
                raise MissingPropertyException('score', k)
            if not isinstance(v['score'], float):
                raise InvalidPropertyException('score', v['score'], k)

    gt_uids = [x['uid'] for x in annotations['annotations']]
    res_uids = list(results['results'].keys())

    missing_uids = np.setdiff1d(gt_uids, res_uids)

    if len(missing_uids) > 0:
        raise MissingUidException(missing_uids)


parser = ArgumentParser()
parser.add_argument('path_to_results_json')
parser.add_argument('path_to_annotations_json')

args = parser.parse_args()

with open(args.path_to_results_json, 'r') as f:
    results = json.load(f)

with open(args.path_to_annotations_json, 'r') as f:
    annotations = json.load(f)

validate_results(results, annotations)

# map = STAMeanAveragePrecision(top_k=5)
map = OverallMeanAveragePrecision(top_k=5)
c = 0
for ann in tqdm(annotations['annotations']):
    uid = ann['uid']

    gt = {
        'boxes': np.vstack([x['box'] for x in ann['objects']]),
        'nouns': np.array([x['noun_category_id'] for x in ann['objects']]),
        'verbs': np.array([x['verb_category_id'] for x in ann['objects']]),
        'ttcs': np.array([x['time_to_contact'] for x in ann['objects']])
    }

    prediction = results['results'][uid]

    if len(prediction) > 0:
        pred = {
            'boxes': np.vstack([x['box'] for x in prediction]),
            'nouns': np.array([x['noun_category_id'] for x in prediction]),
            'verbs': np.array([x['verb_category_id'] for x in prediction]),
            'ttcs': np.array([x['time_to_contact'] for x in prediction]),
            'scores': np.array([x['score'] for x in prediction])
        }
    else:
        pred = {}

    map.add(pred, gt)

scores = map.evaluate()

names = map.get_names()

names[-1] = "* " + names[-1]

for name, val in zip(names, scores):
    print(f"{name}: {val:0.2f}")

print('* metric used to score submissions for the challenge')
