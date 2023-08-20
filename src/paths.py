"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : Paths for general items in the project.
"""
import os

LISTS_DIRECTORY = "data/lists"

DATA_DIRECTORY = "data"

PDB_DIRECTORY = "data/pdb"
PDB_TEXT_DIRECTORY = os.path.join(PDB_DIRECTORY, "pdb_text")
PDB_RAW_DIRECTORY = os.path.join(PDB_DIRECTORY, "pdb_raw")

MODEL_DIRECTORY = "models"
EMBEDDING_MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, "embedding_models")
EXPERIMENT_DIRECTORY = "experiments"
LOG_DIRECTORY = os.path.join(MODEL_DIRECTORY, "logs")

EMBEDDING_DIRECTORY = os.path.join(DATA_DIRECTORY, "embeddings")
DMASIF_EMBEDDING_DIRECTORY = os.path.join(EMBEDDING_DIRECTORY, "dmasif_embeddings")
DISTANCE_MATRICES_DIRECTORY = os.path.join(DATA_DIRECTORY, "distance_matrices")


