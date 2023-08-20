from src.paths import *
from src.utilities import *
from src.experiments.ppi_helpers.ppi_functions import *


def make_dmasif_ppi_predictions(type="max"):
    dummy_list = read_list(os.path.join(LISTS_DIRECTORY, "dummy_id.txt"))
    list_file_path = os.path.join(LISTS_DIRECTORY, "test.txt")
    pdb_list = read_list(list_file_path)

    subject_list = [i.split("_")[0] + "_" + i.split("_")[1] for i in pdb_list if len(i.split("_")) == 3]

    scores_array = np.load(os.path.join(DISTANCE_MATRICES_DIRECTORY, f"{type}_scores.npy"))
    scores_array = abs(scores_array)

    diagonal = np.diagonal(scores_array)
    test_scores = remove_diag(scores_array)
    pairs = len(subject_list)
    idx = [i for i in range(int(pairs))]

    probabilities, predictions, labels, metrics_dict = make_predictions(idx=idx,
                                                                        distances=test_scores,
                                                                        diagonal=diagonal,
                                                                        dummy_list=dummy_list)


    return probabilities, predictions, labels

# make_dmasif_ppi_predictions()