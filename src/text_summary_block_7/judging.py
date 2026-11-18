"""Select judges once; both evaluate identical rows, independently."""
from .data import judge_sample
from .judge import evaluate_judge, settings_hash


def evaluation_plan(pred, config, selected):
    sample = judge_sample(pred, config["judge"]["sample_size"], config["judge"]["seed"])
    return sample, {"sample_ids": sample.id.tolist(), "judges": {
        alias: {"model": settings["model"], "hash": settings_hash(settings)} for alias, settings in selected.items()}}


def evaluate_judges(pred, selected, directory, sample, response_attempts=3):
    return sum(evaluate_judge(pred, settings, directory, alias=alias, sample=sample, response_attempts=response_attempts) for alias, settings in selected.items())
