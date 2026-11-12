"""Sentence metrics and a separate corpus BLEU (never mean sentence BLEU)."""
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from nltk.tokenize import sent_tokenize
from .artifacts import METRICS


def compute_metrics(pred, settings):
    scorer = RougeScorer(METRICS[1:], use_stemmer=settings["use_stemmer"])
    sentence_bleu = BLEU(tokenize=settings["tokenize"], smooth_method=settings["smooth_method"], effective_order=True)
    corpus_bleu = BLEU(tokenize=settings["tokenize"], smooth_method=settings["smooth_method"], effective_order=False)
    rows = []
    for row in pred.itertuples():
        scores = scorer.score(row.reference, row.prediction)
        # ROUGE-Lsum has sentence boundaries; ordinary ROUGE sees original text.
        lsum = scorer.score("\n".join(sent_tokenize(row.reference)), "\n".join(sent_tokenize(row.prediction)))["rougeLsum"].fmeasure
        rows.append({"id": row.id, "model": row.model, "run_id": row.run_id, "response_hash": row.response_hash,
                     "bleu": sentence_bleu.sentence_score(row.prediction, [row.reference]).score,
                     **{k: v.fmeasure for k, v in scores.items()}, "rougeLsum": lsum})
    metrics = pd.DataFrame(rows)
    summary = {"model": pred.model.iloc[0], "run_id": pred.run_id.iloc[0], "count": len(pred),
               "corpus_bleu": corpus_bleu.corpus_score(pred.prediction.tolist(), [pred.reference.tolist()]).score,
               **metrics[METRICS].mean().to_dict()}
    return metrics, pd.DataFrame([summary])
