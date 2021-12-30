from features.feature.binary import binary
from features.feature import Pse
from features.pubscripts.read_fasta_sequences import read_nucleotide_sequences
from features import check_parameters
from features.feature.CKSNAP import CKSNAP
from features.feature.ENAC import ENAC
from features.feature.ANF import ANF
from features.feature.NCP import NCP
import pandas as pd
filename = './M5C_sequences.txt'
fasta = read_nucleotide_sequences(filename)
my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(fasta)

PseDNC_init = pd.DataFrame(Pse.make_SCPseDNC_vector(fasta, my_property_name, my_property_value, lamada_value, weight)).iloc[1:,2:]
binary_init = pd.DataFrame(binary(fasta)).iloc[1:,2:]
CKSNAP_init = pd.DataFrame(CKSNAP(fasta)).iloc[1:,2:]
ENAC_init = pd.DataFrame(ENAC(fasta)).iloc[1:,2:]
ANF_init = pd.DataFrame(ANF(fasta)).iloc[1:,2:]
NCP_init =pd.DataFrame(NCP(fasta)).iloc[1:,2:]

print(binary_init)
print(ENAC_init)
print(ANF_init)
print(NCP_init)
print(PseDNC_init)
print(CKSNAP_init)


