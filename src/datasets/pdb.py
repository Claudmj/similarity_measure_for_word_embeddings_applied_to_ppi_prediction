"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""

from src.paths import *
import os
from Bio.PDB import PDBParser

class PDB:
    """
    Extracts the protein sequence from the PDB file.

    Args:
        protein_code: str
    """
    def __init__(self, protein_code=None):
        self.protein_id = protein_code[0:4]
        self.chain_id = protein_code[5:]
        self.file_name_pdb = os.path.join(PDB_RAW_DIRECTORY, protein_code + ".pdb")
        self.file_name_txt = os.path.join(PDB_DIRECTORY, "pdb_sequence/" + protein_code + ".txt")

        self.num_chains = 0
        self.num_residues = 0
        self.num_atoms = 0

        PARSER_QUIET = True

        if os.path.isfile(self.file_name_txt):
            self.pdb_text = self.load_pdb_text()
        else:
            parser = PDBParser(QUIET=PARSER_QUIET)
            structure = parser.get_structure("structure", os.path.join(PDB_RAW_DIRECTORY, protein_code + ".pdb"))
            self.get_sequence(structure)
            self.save_pdb_text()


    def get_sequence(self, structure):
        self.chain_names = []
        self.pdb_text_list = []


        for model in structure.get_models():

            for chain in model.get_chains():
                self.chain_names.append(chain.id)

                for residue in chain.get_residues():
                    self.pdb_text_list.append(residue.resname)

        self.pdb_text = " ".join(self.pdb_text_list)

    def save_pdb_text(self):
        with open(self.file_name_txt, 'w') as file:
            file.write(self.pdb_text)
        file.close()


    def load_pdb_text(self):
        with open(self.file_name_txt, "r") as file:
            lines = file.read()
            return lines