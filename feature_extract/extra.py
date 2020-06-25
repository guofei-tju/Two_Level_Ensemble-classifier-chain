import test_for_paper2.DWT as dwt
import test_for_paper2.AvBlock as avb
import test_for_paper2.PseudoPSSM as pse
import test_for_paper2.ASDC as asdc


def extra_DWT(protein):
    return dwt.GetDWT(protein)

def extra_avb(protein):
    return avb.GetAvBlock(protein)

def extra_pse(protein):
    return pse.GetPse(protein,30)

def extra_asdc(protein):
    return asdc.GetASDC(protein)
