import numpy as np
from abc import ABC, abstractmethod


def compute_iou(preds, gts):
    """
    Compute a matrix of intersection over union values for two lists of bounding boxes using broadcasting
    :param preds: matrix of predicted bounding boxes [NP x 4]
    :param gts: number of ground truth bounding boxes [NG x 4]
    :return: an [NP x NG] matrix of IOU values
    """
    # Convert shapes to use broadcasting
    # preds: NP x 4 -> NP x 1 x 4
    # gts: NG x 4 -> 1 x NG x 4
    preds = np.expand_dims(preds, 1)
    gts = np.expand_dims(gts, 0)

    def area(boxes):
        width = boxes[..., 2] - boxes[..., 0] + 1
        height = boxes[..., 3] - boxes[..., 1] + 1
        width[width < 0] = 0
        height[height < 0] = 0
        return width * height

    ixmin = np.maximum(gts[..., 0], preds[..., 0])
    iymin = np.maximum(gts[..., 1], preds[..., 1])
    ixmax = np.minimum(gts[..., 2], preds[..., 2])
    iymax = np.minimum(gts[..., 3], preds[..., 3])

    areas_preds = area(preds)
    areas_gts = area(gts)
    areas_intersections = area(np.stack([ixmin, iymin, ixmax, iymax], -1))

    return (areas_intersections) / (areas_preds + areas_gts - areas_intersections + 1e-11)


class AbstractMeanAveragePrecision(ABC):
    """
    Abstract class for implementing mAP measures
    """

    def __init__(self, num_aps, percentages=True, count_all_classes=True, top_k=None):
        """
        Contruct the Mean Average Precision metric
        :param num_aps: number of average precision metrics to compute. E.g., we can compute different APs for different
                        IOU overlap thresholds
        :param percentages: whether the metric should return percentages (i.e., 0-100 range rather than 0-1)
        :param count_all_classes: whether to count all classes when computing mAP. If false, classes which do not have
                                    any ground truth label but do have associated predictions are counted (they will have
                                    an AP equal to zero), otherwise, only classes for which there is at least one ground truth
                                    label will count. It is useful to set this to True for imbalanced datasets for which not
                                    all classes are in the ground truth labels.
        :param top_k: the K to be considered in the top-k criterion. If None, a standard mAP will be computed
        """
        self.true_positives = []
        self.confidence_scores = []
        self.predicted_classes = []
        self.gt_classes = []

        self.num_aps = num_aps
        self.percentages = percentages
        self.count_all_classes = count_all_classes
        self.K = top_k
        self.names = []
        self.short_names = []

    def get_names(self):
        return self.names

    def get_short_names(self):
        return self.short_names

    def add(self,
            preds,
            labels
            ):
        """
        Add predictions and labels of a single image and matches predictions to ground truth boxes
        :param predictions: dictionary of predictions following the format below. While "boxes" and "scores" are
                            mandatory, other properties can be added (they can be used to compute matchings).
                            It can also be a list of dictionaries if predictions of more than one images are being added.
                {
                    'boxes' : [
                        [245,128,589,683],
                        [425,68,592,128]
                    ],
                    'scores' : [
                        0.8,
                        0.4
                    ],
                    'nouns' : [
                        3,
                        5
                    ],
                    'verbs': [
                        8,
                        11
                    ],
                    'ttcs': [
                        1.25,
                        1.8
                    ]
                }
        :param labels: dictionary of labels following a similar format. It can be a list of dictionaries.
                {
                    'boxes' : [
                        [195,322,625,800],
                        [150,300,425,689]
                    ],
                    'nouns' : [
                        9,
                        5
                    ],
                    'verbs': [
                        3,
                        11
                    ],
                    'ttcs': [
                        0.25,
                        1.25
                    ]
                }
        :return matched: a list of pairs of predicted/matched gt boxes
        """
        matched = []

        if len(preds) > 0:
            predicted_boxes = preds['boxes']
            predicted_scores = preds['scores']
            predicted_classes = self._map_classes(preds)

            # Keep track of correctly matched boxes for the different AP metrics
            true_positives = np.zeros((len(predicted_boxes), self.num_aps))

            if len(labels) > 0:
                # get GT boxes
                gt_boxes = labels['boxes']

                # IOU between all predictions and gt boxes
                ious = compute_iou(predicted_boxes, gt_boxes)

                # keep track of GT boxes which have already been matched
                gt_matched = np.zeros((len(gt_boxes), self.num_aps))

                # from highest to lowest score
                for i in predicted_scores.argsort()[::-1]:
                    # get overlaps related to this prediction
                    overlaps = ious[i].reshape(-1, 1)  # NGT x 1

                    # check if this prediction can be matched to the GT labels
                    # this will give different set of matchings for the different AP metrics
                    matchings = self._match({k: p[i] for k, p in preds.items()}, labels, overlaps)  # NGT x NR

                    # replicate overlaps to match shape of matching (different AP metrics)
                    overlaps = np.tile(overlaps, [1, matchings.shape[1]])  # NGT x NR

                    # do not allow to match a matched GT boxes
                    matchings[gt_matched == 1] = 0  # not a valid match #NGT x NR

                    # remove overlaps corresponding to boxes which are not a match
                    overlaps[matchings == 0] = -1

                    jj = overlaps.argmax(0)  # get indexes of maxima wrt GT

                    # get values of matching obtained at maxima
                    # these indicate if the matchings are correct
                    i_matchings = matchings[jj, range(len(jj))]

                    jj_matched = jj.copy()
                    jj_matched[~i_matchings] = -1

                    # set true positive to 1 if we obtained a matching
                    true_positives[i, i_matchings] = 1

                    # set the ground truth as matched if we obtained a matching
                    gt_matched[jj, range(len(jj))] += i_matchings

                    matched.append(jj_matched)

                # remove the K highest score false positives
                if self.K is not None and self.K > 1:
                    # number of FP to remove:
                    K = (self.K - 1) * len(labels['boxes'])
                    # indexes to sort the predictions
                    order = predicted_scores.argsort()[::-1]
                    # sort the true positives labels
                    sorted_tp = (true_positives[order, :]).astype(float)
                    # invert to obtain the sorted false positive labels
                    sorted_fp = 1 - sorted_tp
                    # flag the first K false positives
                    sorted_tp[(sorted_fp.cumsum(0) <= K) & (sorted_fp == 1)] = np.nan

                    true_positives = sorted_tp
                    predicted_scores = predicted_scores[order]
                    predicted_classes = predicted_classes[order]

                self.gt_classes.append(self._map_classes(labels))

            # append list of true positives and confidence scores
            self.true_positives.append(true_positives)
            self.confidence_scores.append(predicted_scores)
            self.predicted_classes.append(predicted_classes)
        else:
            if len(preds) > 0:
                self.gt_classes.append(self._map_classes(labels))
        if len(matched) > 0:
            return np.stack(matched, 0)
        else:
            return np.zeros((0, self.num_aps))

    def _map_classes(self, preds):
        """
        Return the classes related to the predictions. These are used to specify how to compute mAP.
        :param preds: the predictions
        :return: num_ap x len(pred) array specifying the class of each prediction according to the different AP measures
        """
        return np.vstack([preds['nouns']] * self.num_aps).T

    def _compute_prec_rec(self, true_positives, confidence_scores, num_gt):
        """
        Compute precision and recall curve from a true positive list and the related scores
        :param true_positives: set of true positives
        :param confidence_scores:  scores associated to the true positives
        :param num_gt: number of ground truth labels for current class
        :return: prec, rec: lists of precisions and recalls
        """
        # sort true positives by confidence score
        tps = true_positives[confidence_scores.argsort()[::-1]]

        tp = tps.cumsum()
        fp = (1 - tps).cumsum()

        # safe division which turns x/0 to zero
        prec = self._safe_division(tp, tp + fp)
        rec = self._safe_division(tp, num_gt)

        return prec, rec

    def _safe_division(self, a, b):
        """
        Divide a by b avoiding a DivideByZero exception
        Inputs:
            a, b: either vectors or scalars
        Outputs:
            either a vector or a scalar
        """
        a_array = isinstance(a, np.ndarray)
        b_array = isinstance(b, np.ndarray)

        if (not a_array) and (not b_array):
            # both scalars
            # anything divided by zero should be zero
            if b == 0:
                return 0

        # numerator scalar, denominator vector
        if b_array and not a_array:
            # turn a into a vector
            a = np.array([a] * len(b))

        # numerator vector, denominator scalar
        if not b_array and a_array:
            # turn a into a vector
            b = np.array([b] * len(a))

        # turn all cases in which b=0 in a 0/1 division (result is 0)
        zeroden = b == 0
        b[zeroden] = 1
        a[zeroden] = 0
        return a / b

    def _compute_ap(self, prec, rec):
        """
        Python implementation of Matlab VOC AP code.
            1) Make precision monotonically decreasing 2) tThen compute AP by numerical integration.
        :param prec: vector of precision values
        :param rec: vector of recall values
        :return: average precision
        """
        # pad precision and recall
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.concatenate(([0], prec, [0]))

        # make precision monotonically decresing
        for i in range(len(mpre) - 2, 0, -1):
            mpre[i] = np.max((mpre[i], mpre[i + 1]))

        # consider only indexes in which the recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1

        # compute the area uner the curve
        return np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    def _compute_mr(self, prec, rec):
        """
        Compute maximum recall
        """
        return np.max(rec)

    def evaluate(self, measure='AP'):
        """
        Compute AP/MR for all classes, then averages
        """

        metrics = []
        # compute the different AP values for the different metrics

        gt_classes = np.concatenate(self.gt_classes)
        predicted_classes = np.concatenate(self.predicted_classes)
        true_positives = np.concatenate(self.true_positives)
        confidence_scores = np.concatenate(self.confidence_scores)

        for i in range(self.num_aps):
            # the different per-class AP values
            measures = []

            _gt_classes = gt_classes[:, i]
            _predicted_classes = predicted_classes[:, i]
            _true_positives = true_positives[:, i]
            _confidence_scores = confidence_scores

            if self.count_all_classes:
                classes = np.unique(np.concatenate([_gt_classes, _predicted_classes]))
            else:
                classes = np.unique(_gt_classes)

            # iterate over classes
            for c in classes:
                # get true positives and number of GT values
                tp = _true_positives[_predicted_classes == c]
                cs = _confidence_scores[_predicted_classes == c]
                ngt = np.sum(_gt_classes == c)

                # check if the list of TP is non empty
                if len(tp) > 0:
                    # remove invalid TP values and related confidence scores
                    valid = ~np.isnan(tp)
                    tp, cs = tp[valid], cs[valid]
                # if both TP and GT are non empty, then compute AP
                if len(tp) > 0 and ngt > 0:
                    prec, rec = self._compute_prec_rec(tp, cs, ngt)
                    if measure == 'AP':
                        this_measure = self._compute_ap(prec, rec)
                    elif measure == 'MR':  # maximum recall
                        this_measure = self._compute_mr(prec, rec)
                    # turn into percentage
                    if self.percentages:
                        this_measure = this_measure * 100
                    # append to the list
                    measures.append(this_measure)
                # if both are empty, the AP is zero
                elif not (len(tp) == 0 and ngt == 0):
                    measures.append(0)
            # append the mAP value
            metrics.append(np.mean(measures))

        # return single value or list of values
        values = list(metrics)
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    @abstractmethod
    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT labels
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """


class ObjectOnlyMeanAveragePrecision(AbstractMeanAveragePrecision):
    def __init__(self, iou_threshold=0.5, top_k=3, count_all_classes=False):
        """
        Construct the object only mAP metric. This will compute the following metrics:
            - Box + Noun
            - Box
        :param iou_threshold:
        :param tti_threshold:
        :param top_k:
        :param count_all_classes:
        """
        super().__init__(2, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.names = ["Box + Noun mAP", "Box AP"]
        self.short_names = ["map_box_noun", "ap_box"]

    def _map_classes(self, preds):
        """
        Associates each prediction to a class
        :param preds: the input predictions
        :return the matrix of classess associated to each prediction according to the evaluation measure
        """
        nouns = preds['nouns']
        boxes = np.ones(len(preds['nouns']))

        return np.vstack([
            nouns,  # box + noun, average over nouns
            boxes]  # box, just compute a single AP
        ).T

    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT predictions
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """
        nouns = (pred['nouns'] == gt_predictions['nouns'])
        boxes = (ious.ravel() > self.iou_threshold)

        map_box_noun = boxes & nouns
        map_box = boxes

        return np.vstack([map_box_noun, map_box]).T


class OverallMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Compute the different STA metrics based on mAP"""

    def __init__(self, iou_threshold=0.5, ttc_threshold=0.25, top_k=5, count_all_classes=False):
        """
        Construct the overall mAP metric. This will compute the following metrics:
            - Box AP
            - Box + Noun AP
            - Box + Verb AP
            - Box + TTC AP
            - Box + Verb + TTC AP
            - Box + Noun mAP
            - Box + Noun + Verb mAP
            - Box + Noun + TTC mAP
            - Box + Noun + Verb + TTC mAP
        :param iou_threshold: IOU threshold to check if a predicted box can be matched to a ground turth box
        :param ttc_threshold: TTC threshold to check if a predicted TTC is acceptable
        :param top_k: Top-K criterion for mAP. Discounts up to k-1 high scoring false positives
        :param count_all_classes: whether to also average across classes with no annotations. False is the default for many implementations.
        """
        super().__init__(12, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.tti_threshold = ttc_threshold

        self.names = ['Box AP',
                      'Box + Noun AP',
                      'Box + Verb AP',
                      'Box + TTC AP',
                      'Box + Noun + Verb AP',
                      'Box + Noun + TTC AP',
                      'Box + Verb + TTC AP',
                      'Box + Noun + Verb + TTC AP',
                      'Box + Noun mAP',
                      'Box + Noun + Verb mAP',
                      'Box + Noun + TTC mAP',
                      'Box + Noun + Verb + TTC mAP']

        self.short_names = ['ap_box',
                            'ap_box_noun',
                            'ap_box_verb',
                            'ap_box_ttc',
                            'ap_box_noun_verb',
                            'ap_box_noun_ttc',
                            'ap_box_verb_ttc',
                            'ap_box_noun_verb_ttc',
                            'map_box_noun',
                            'map_box_noun_verb',
                            'map_box_noun_ttc',
                            'map_box_noun_verb_ttc']

    def _map_classes(self, preds):
        """
        Associates each prediction to a class
        :param preds: the input predictions
        :return the matrix of classess associated to each prediction according to the evaluation measure
        """
        nouns = preds['nouns']
        ones = np.ones(len(preds['nouns']))

        return np.vstack([
            ones,  # ap_box - do not average
            ones,  # ap_box_noun - do not average
            ones,  # ap_box_verb - do not average
            ones,  # ap_box_ttc - do not average
            ones,  # ap_box_noun_verb - do not average
            ones,  # ap_box_noun_ttc - do not average
            ones,  # ap_box_verb_ttc - do not average
            ones,  # ap_box_noun_verb_ttc - do not average
            nouns,  # map_box_noun - average over nouns
            nouns,  # map_box_noun_verb - average over nouns
            nouns,  # map_box_noun_ttc - average over nouns
            nouns  # map_box_noun_verb_ttc - average over nouns
        ]).T

    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT predictions
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """
        nouns = (pred['nouns'] == gt_predictions['nouns'])
        boxes = (ious.ravel() > self.iou_threshold)
        verbs = (pred['verbs'] == gt_predictions['verbs'])
        ttcs = (np.abs(pred['ttcs'] - gt_predictions['ttcs']) <= self.tti_threshold)

        tp_box = boxes
        tp_box_noun = boxes & nouns
        tp_box_verb = boxes & verbs
        tp_box_ttc = boxes & ttcs
        tp_box_noun_verb = boxes & verbs & nouns
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_verb_ttc = boxes & verbs & ttcs
        tp_box_noun_verb_ttc = boxes & verbs & nouns & ttcs

        return np.vstack([tp_box,  # ap_box
                          tp_box_noun,  # ap_box_noun
                          tp_box_verb,  # ap_box_verb
                          tp_box_ttc,  # ap_box_ttc
                          tp_box_noun_verb,  # ap_box_noun_verb
                          tp_box_noun_ttc,  # ap_box_noun_ttc
                          tp_box_verb_ttc,  # ap_box_verb_ttc
                          tp_box_noun_verb_ttc,  # ap_box_noun_verb_ttc
                          tp_box_noun,  # map_box_noun
                          tp_box_noun_verb,  # map_box_noun_verb
                          tp_box_noun_ttc,  # map_box_noun_ttc
                          tp_box_noun_verb_ttc  # map_box_noun_verb_ttc
                          ]).T


class STAMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Compute the different STA metrics based on mAP"""

    def __init__(self, iou_threshold=0.5, ttc_threshold=0.25, top_k=5, count_all_classes=False):
        """
        Construct the overall mAP metric. This will compute the following metrics:
            - Box + Noun mAP
            - Box + Noun + Verb mAP
            - Box + Noun + TTC mAP
            - Box + Noun + Verb + TTC mAP
        :param iou_threshold: IOU threshold to check if a predicted box can be matched to a ground turth box
        :param ttc_threshold: TTC threshold to check if a predicted TTC is acceptable
        :param top_k: Top-K criterion for mAP. Discounts up to k-1 high scoring false positives
        :param count_all_classes: whether to also average across classes with no annotations. False is the default for many implementations.
        """
        super().__init__(4, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.tti_threshold = ttc_threshold

        self.names = ['Box AP',
                      'Box + Noun mAP',
                      'Box + Noun + Verb mAP',
                      'Box + Noun + TTC mAP',
                      'Box + Noun + Verb + TTC mAP']

        self.short_names = ['ap_box',
                            'map_box_noun',
                            'map_box_noun_verb',
                            'map_box_noun_ttc',
                            'map_box_noun_verb_ttc']

    def _map_classes(self, preds):
        """
        Associates each prediction to a class
        :param preds: the input predictions
        :return the matrix of classess associated to each prediction according to the evaluation measure
        """
        nouns = preds['nouns']

        return np.vstack([
            nouns,  # map_box_noun - average over nouns
            nouns,  # map_box_noun_verb - average over nouns
            nouns,  # map_box_noun_ttc - average over nouns
            nouns  # map_box_noun_verb_ttc - average over nouns
        ]).T

    def _match(self, pred, gt_predictions, ious):
        """
        Return matches of a given prediction to a set of GT predictions
        :param pred: the prediction dictionary
        :param gt_predictions: the gt predictions dictionary
        :param ious: the computed IOU matrix (NGT x NPRED)
        :return: a num_preds x num_ap matrix specifying possible matchings depending on the prediction and metric
        """
        nouns = (pred['nouns'] == gt_predictions['nouns'])
        boxes = (ious.ravel() > self.iou_threshold)
        verbs = (pred['verbs'] == gt_predictions['verbs'])
        ttcs = (np.abs(pred['ttcs'] - gt_predictions['ttcs']) <= self.tti_threshold)

        tp_box_noun = boxes & nouns
        tp_box_noun_verb = boxes & verbs & nouns
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_noun_verb_ttc = boxes & verbs & nouns & ttcs

        return np.vstack([tp_box_noun,  # map_box_noun
                          tp_box_noun_verb,  # map_box_noun_verb
                          tp_box_noun_ttc,  # map_box_noun_ttc
                          tp_box_noun_verb_ttc  # map_box_noun_verb_ttc
                          ]).T
