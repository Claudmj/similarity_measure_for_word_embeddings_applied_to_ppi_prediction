import matplotlib.pyplot as plt
from sklearn import metrics

from src.paths import *
from src.experiments.ppi_helpers.ppi_prediction_lda import make_ppi_predictions_lda
from src.experiments.ppi_helpers.ppi_dmasif_prediction import make_dmasif_ppi_predictions
from src.experiments.ppi_helpers.ppi_prediction_word2vec import make_ppi_predictions_word2vec_cosine, make_ppi_predictions_word2vec_sc

probs_lda, preds_lda, labels = make_ppi_predictions_lda()
probs_word2vec_sc, preds_word2vec_sc, labels = make_ppi_predictions_word2vec_sc()
probs_word2vec_cosine, preds_word2vec_cosine, labels = make_ppi_predictions_word2vec_cosine()
probs_dmasif_max, preds_dmasif_max, labels = make_dmasif_ppi_predictions(type="max")
probs_dmasif_mean, preds_dmasif_mean, labels = make_dmasif_ppi_predictions(type="mean")


plt.figure(figsize=(7, 7))
plt.title("ROC Curves")
plt.xlabel("FPR")
plt.ylabel("TPR")
fpr_lda, tpr_lda, _ = metrics.roc_curve(labels,  probs_lda)
fpr_word2vec_sc, tpr_word2vec_sc, _ = metrics.roc_curve(labels,  probs_word2vec_sc)
fpr_word2vec_cosine, tpr_word2vec_cosine, _ = metrics.roc_curve(labels,  probs_word2vec_cosine)
fpr_dmasif_max, tpr_dmasif_max, _ = metrics.roc_curve(labels,  probs_dmasif_max)
fpr_dmasif_mean, tpr_dmasif_mean, _ = metrics.roc_curve(labels,  probs_dmasif_mean)
plt.plot(fpr_lda, tpr_lda, label=f"LDA and BD AUC: {metrics.roc_auc_score(labels, probs_lda): .4f}")
plt.plot(fpr_word2vec_sc, tpr_word2vec_sc, label=f"word2vec and SCS AUC: {metrics.roc_auc_score(labels, probs_word2vec_sc): .4f}")
plt.plot(fpr_word2vec_cosine, tpr_word2vec_cosine, label=f"word2vec and CS AUC: {metrics.roc_auc_score(labels, probs_word2vec_cosine): .4f}")
plt.plot(fpr_dmasif_max, tpr_dmasif_max, label=f"dMaSIF max AUC: {metrics.roc_auc_score(labels, probs_dmasif_max): .4f}")
plt.plot(fpr_dmasif_mean, tpr_dmasif_mean, label=f"dMaSIF mean AUC: {metrics.roc_auc_score(labels, probs_dmasif_mean): .4f}")
plt.legend(loc=4)
plt.savefig(os.path.join(LOG_DIRECTORY, "ppi_prediction.png"), format='png', dpi=120)
plt.show()