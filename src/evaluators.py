from dspy import Example
from dspy.evaluate import Evaluate
from src.metrics import *
import os

num_threads = os.environ.get('DSP_NUM_THREADS', 1)

""" Wrap metrics in an interface to be used by DSPy and create evaluators to easily run these metrics across a set of examples.
"""


# wrap the metrics for use in DSPy
def dspy_metric_rp50(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 50)


def dspy_metric_rp10(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 10)


def dspy_metric_rp5(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 5)


def dspy_metric_rp1(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 1)


def dspy_metric_recall10(gold: Example, pred, trace=None) -> float:
    return recall_at_k(gold.label, pred.predictions, 10)


def dspy_metric_recall5(gold: Example, pred, trace=None) -> float:
    return recall_at_k(gold.label, pred.predictions, 5)


def dspy_metric_recall1(gold: Example, pred, trace=None) -> float:
    return recall_at_k(gold.label, pred.predictions, 1)

def dspy_metric_entity_infer_custom_metric_at_1(gold: Example, pred, trace=None) -> float:
    return entity_infer_custom_metric_at_k(gold.label, pred.predictions, 1)


def create_evaluators(examples):
    # create a suite of DSPy evaluators based on a set of examples
    evaluate_recall10 = Evaluate(
        devset=examples,
        metric=dspy_metric_recall10,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_recall5 = Evaluate(
        devset=examples,
        metric=dspy_metric_recall5,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_recall1 = Evaluate(
        devset=examples,
        metric=dspy_metric_recall1,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp50 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp50,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp10 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp10,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp5 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp5,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp1 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp1,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    return {
        "recall10": evaluate_recall10,
        "recall5": evaluate_recall5,
        "recall1": evaluate_recall1,
        "rp50": evaluate_rp50,
        "rp10": evaluate_rp10,
        "rp5": evaluate_rp5,
        "rp1": evaluate_rp1,
    }


supported_metrics = {
    "rp1": dspy_metric_rp1,
    "rp5": dspy_metric_rp5,
    "rp10": dspy_metric_rp10,
    "rp50": dspy_metric_rp50,
    "recall1": dspy_metric_recall1,
    "recall5": dspy_metric_recall5,
    "recall10": dspy_metric_recall10,
    "infer1": dspy_metric_entity_infer_custom_metric_at_1,
}
