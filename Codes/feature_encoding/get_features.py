import pandas as pd
import os

os.path.join('~/m5C-model')
from features.feature.binary import binary
from features.feature import Pse
from features.pubscripts.read_fasta_sequences import read_nucleotide_sequences
from features import check_parameters
from features.feature.CKSNAP import CKSNAP
from features.feature.ENAC import ENAC
from features.feature.ANF import ANF
from features.feature.NCP import NCP


def get_fasttext(filename,sp):

    sp = '../model_fast/' + sp + '/nsv_6gram_model.bin'
    cmd_1 = 'python3 ../fasttext_generated_ngram_non_sup.py ' + filename + ' ' + filename+'.txt 6'

    cmd_2 = 'cat '+filename+'.txt | ../fastText-0.9.2/fasttext print-sentence-vectors ' + sp
    print(cmd_1)
    print(cmd_2)
    p1 = os.popen(cmd_1,'r')
    print(p1.read())
    p2 = os.popen(cmd_2,'r')
    fasttext = p2.read()

    return fasttext

def Get(filename,sp):
    fasta = read_nucleotide_sequences(filename)
    my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(fasta)

    PseDNC_init = pd.DataFrame(Pse.make_SCPseDNC_vector(fasta, my_property_name, my_property_value, lamada_value, weight)).iloc[1:,2:]
    binary_init = pd.DataFrame(binary(fasta)).iloc[1:,2:]
    CKSNAP_init = pd.DataFrame(CKSNAP(fasta)).iloc[1:,2:]
    ENAC_init = pd.DataFrame(ENAC(fasta)).iloc[1:,2:]
    ANF_init = pd.DataFrame(ANF(fasta)).iloc[1:,2:]
    NCP_init =pd.DataFrame(NCP(fasta)).iloc[1:,2:]

    features_all = pd.concat([binary_init, ENAC_init, ANF_init, NCP_init, PseDNC_init, CKSNAP_init], axis=1, ignore_index=True)
    print(features_all)
    return features_all


if __name__ == '__main__':
    get_fasttext('./M5C_sequences.txt', 'AT')
