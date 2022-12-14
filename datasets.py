from forecasting_eval.ego4d.datasets.short_term_anticipation import Ego4dShortTermAnticipation
def build_short_term_dataset(cfg,split):
    return Ego4dShortTermAnticipation(cfg=cfg,split=split)