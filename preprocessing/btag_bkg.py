import uproot
import awkward as ak
from glob import glob
from coffea.nanoevents import NanoEventsFactory, DelphesSchema, BaseSchema
import pickle, json, gzip
import numpy as np
from typing import Optional, List, Dict, Tuple
from copy import copy
from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import boost_histogram as bh
from cycler import cycler
from tqdm import tqdm
import pathlib
import os
import boost_histogram as bh

from optparse import OptionParser
'''
This file is for BKG
'''
plot_dir = "../plots/tagger/18Aug2024"
_ = os.system(f"mkdir -p {plot_dir}")

import custom
print("Loaded CustomDelphesSchema:", custom.CustomDelphesSchema)

parser = OptionParser()
parser.add_option('--infile', help='input  delphes root file')
parser.add_option('--outfile', help='input  delphes root file')
(options, args) = parser.parse_args()



delphes_roots = {
    'TTbar_semilep' : options.infile,
}

files = {typefile : {} for typefile in delphes_roots}
for typefile in delphes_roots:
    # files[typefile] = uproot.lazy(delphes_roots[typefile])
    files[typefile] = NanoEventsFactory.from_root(
        delphes_roots[typefile],
        treepath="/tree",
        schemaclass=custom.CustomDelphesSchema,
    ).events()


labels = [
    #  H->2 prong
    "H_bb", "H_cc", "H_ss", "H_qq", "H_bc", "H_cs", "H_bq", "H_cq", "H_sq", "H_gg", "H_ee", "H_mm", "H_tauhtaue", "H_tauhtaum", "H_tauhtauh", 

    #  H->4/3 prong
    "H_AA_bbbb", "H_AA_bbcc", "H_AA_bbss", "H_AA_bbqq", "H_AA_bbgg", "H_AA_bbee", "H_AA_bbmm",
    "H_AA_bbtauhtaue", "H_AA_bbtauhtaum", "H_AA_bbtauhtauh",

    "H_AA_bbb", "H_AA_bbc", "H_AA_bbs", "H_AA_bbq", "H_AA_bbg", "H_AA_bbe", "H_AA_bbm",

    "H_AA_cccc", "H_AA_ccss", "H_AA_ccqq", "H_AA_ccgg", "H_AA_ccee", "H_AA_ccmm",
    "H_AA_cctauhtaue", "H_AA_cctauhtaum", "H_AA_cctauhtauh",

    "H_AA_ccb", "H_AA_ccc", "H_AA_ccs", "H_AA_ccq", "H_AA_ccg", "H_AA_cce", "H_AA_ccm",

    "H_AA_ssss", "H_AA_ssqq", "H_AA_ssgg", "H_AA_ssee", "H_AA_ssmm",
    "H_AA_sstauhtaue", "H_AA_sstauhtaum", "H_AA_sstauhtauh",

    "H_AA_ssb", "H_AA_ssc", "H_AA_sss", "H_AA_ssq", "H_AA_ssg", "H_AA_sse", "H_AA_ssm",

    "H_AA_qqqq", "H_AA_qqgg", "H_AA_qqee", "H_AA_qqmm",
    "H_AA_qqtauhtaue", "H_AA_qqtauhtaum", "H_AA_qqtauhtauh",

    "H_AA_qqb", "H_AA_qqc", "H_AA_qqs", "H_AA_qqq", "H_AA_qqg", "H_AA_qqe", "H_AA_qqm",

    "H_AA_gggg", "H_AA_ggee", "H_AA_ggmm",
    "H_AA_ggtauhtaue", "H_AA_ggtauhtaum", "H_AA_ggtauhtauh",

    "H_AA_ggb", "H_AA_ggc", "H_AA_ggs", "H_AA_ggq", "H_AA_ggg", "H_AA_gge", "H_AA_ggm",

    "H_AA_bee", "H_AA_cee", "H_AA_see", "H_AA_qee", "H_AA_gee",
    "H_AA_bmm", "H_AA_cmm", "H_AA_smm", "H_AA_qmm", "H_AA_gmm",

    "H_AA_btauhtaue", "H_AA_ctauhtaue", "H_AA_stauhtaue", "H_AA_qtauhtaue", "H_AA_gtauhtaue",
    "H_AA_btauhtaum", "H_AA_ctauhtaum", "H_AA_stauhtaum", "H_AA_qtauhtaum", "H_AA_gtauhtaum",
    "H_AA_btauhtauh", "H_AA_ctauhtauh", "H_AA_stauhtauh", "H_AA_qtauhtauh", "H_AA_gtauhtauh",

    #  (H+H-: H_AA_bbcs, H_AA_bbsq, H_AA_ssbc, H_AA_ssbq not available)
    "H_AA_qqqb", "H_AA_qqqc", "H_AA_qqqs",
    "H_AA_bbcq",
    "H_AA_ccbs", "H_AA_ccbq", "H_AA_ccsq",
    "H_AA_sscq",
    "H_AA_qqbc", "H_AA_qqbs", "H_AA_qqcs",
    "H_AA_bcsq",

    "H_AA_bcs", "H_AA_bcq", "H_AA_bsq", "H_AA_csq", 

    "H_AA_bcev", "H_AA_csev", "H_AA_bqev", "H_AA_cqev", "H_AA_sqev", "H_AA_qqev",
    "H_AA_bcmv", "H_AA_csmv", "H_AA_bqmv", "H_AA_cqmv", "H_AA_sqmv", "H_AA_qqmv",
    "H_AA_bctauev", "H_AA_cstauev", "H_AA_bqtauev", "H_AA_cqtauev", "H_AA_sqtauev", "H_AA_qqtauev",
    "H_AA_bctaumv", "H_AA_cstaumv", "H_AA_bqtaumv", "H_AA_cqtaumv", "H_AA_sqtaumv", "H_AA_qqtaumv",
    "H_AA_bctauhv", "H_AA_cstauhv", "H_AA_bqtauhv", "H_AA_cqtauhv", "H_AA_sqtauhv", "H_AA_qqtauhv",


    "QCD_bbccss", "QCD_bbccs", "QCD_bbcc", "QCD_bbcss", "QCD_bbcs", "QCD_bbc", "QCD_bbss", "QCD_bbs", "QCD_bb",
    "QCD_bccss", "QCD_bccs", "QCD_bcc", "QCD_bcss", "QCD_bcs", "QCD_bc", "QCD_bss", "QCD_bs", "QCD_b",
    "QCD_ccss", "QCD_ccs", "QCD_cc", "QCD_css", "QCD_cs", "QCD_c", "QCD_ss", "QCD_s", "QCD_light",

]

index_hbc = labels.index("H_bc")
index_hbq = labels.index("H_bq")
index_hbb = labels.index("H_bb")
index_hcc = labels.index("H_cc")
index_qcd_start = labels.index("QCD_bbccss")

# use leading pT jet first, anyway we will only select one AK8 jet in the end
def pad_val(
    arr: ak.Array,
    target: int,
    value: float,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    return ret.to_numpy() if to_numpy else ret

import math
def delta_r(a, b):
    deta = a.Eta - b.Eta
    dphi = (a.Phi - b.Phi + math.pi) % (2 * math.pi) - math.pi
    return np.hypot(deta, dphi)

use_helvet = False ## true: use helvetica for plots, make sure the system have the font installed
if use_helvet:
    CMShelvet = hep.style.CMS
    CMShelvet['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.style.use(CMShelvet)
else:
    plt.style.use(hep.style.ROOT)

def flow(hist: bh.Histogram, overflow: bool=False, underflow: bool=False):
    h, var = hist.view(flow=(overflow | underflow)).value, hist.view(flow=(overflow | underflow)).variance
    if overflow: 
        # h, var also include underflow bins but in plots usually no underflow data
        # And we've filled None with -999, so we shouldn't show underflow data (mostly from filled None)
        # You have to access the overflow and underflow bins data like below:
        h[-2] += h[-1]; var[-2] += var[-1]
    if underflow:
        h[1] += h[0]; var[1] += var[0]
    if overflow or underflow:
        h, var = h[1:-1], var[1:-1]
    return h, var
    # Return the updated histogram and variance

def error_bar(h, var, type='data'):
    from scipy.interpolate import CubicSpline
    if type == 'data':
        number = h
    elif type == 'mc':  # h = k*N, var = k^2*N, std = k*sqrt(N)
        number = h**2 / var
    else:
        raise ValueError("type should be 'data' or 'mc'! ")
    center = range(11) # Number: 0-10
    up = np.array([1.84, 3.30, 4.64, 5.92, 7.16, 8.38, 9.58, 10.77, 11.95, 13.11, 14.27]) - center
    down = center - np.array([0, 0.17, 0.71, 1.37, 2.09, 2.84, 3.62, 4.42, 5.23, 6.06, 6.89])
    #cs means to create a CubicSpline object
    cs_up = CubicSpline(x=center, y=up)
    cs_down = CubicSpline(x=center, y=down)
    
    Garwood = (number>0)&(number<10)
    poison_error_bar = np.sqrt(number)
    up_error_bar = np.copy(poison_error_bar)
    down_error_bar = np.copy(poison_error_bar)
    up_error_bar[Garwood] = cs_up(number[Garwood])
    down_error_bar[Garwood] = cs_down(number[Garwood])
    if type == 'mc':
        up_error_bar *= var/h
        down_error_bar *= var/h
    up_error_bar [up_error_bar < 0 ] = 0
    down_error_bar [down_error_bar < 0 ] = 0
    return np.array([down_error_bar, up_error_bar])

skim_vars = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}




events = files["TTbar_semilep"]


def get_hbcvsqcd(df):
    df["hbc"] = df[f"jet_probs_{str(index_hbc)}"]
    df["qcd"] = ak.zeros_like(df["hbc"])
    for idx in range(index_qcd_start,188):
        df["qcd"] = df["qcd"] + df[f"jet_probs_{str(idx)}"]
    df["hbcvsqcd"] = df["hbc"]/(df["hbc"] + df["qcd"])



#compute tagger score
leading_tagger = np.max(events[f"jet_probs_{str(index_hbc)}"],axis = 1) #leading tagger score
leading_tagger_indices = (events[f"jet_probs_{str(index_hbc)}"] == leading_tagger)

#order other object
get_hbcvsqcd(events)
leading_hbc = events[f"jet_probs_{str(index_hbc)}"][leading_tagger_indices][:,0]
leading_hbcvsqcd = events["hbcvsqcd"][leading_tagger_indices][:,0]
leading_qcd = events["qcd"][leading_tagger_indices][:,0]
leading_hbq = events[f"jet_probs_{str(index_hbq)}"][leading_tagger_indices][:,0]
#currently top not available

#candidate jet is the jet with leading Hbc tagger
leading_fatjet = events.JetPUPPIAK8[leading_tagger_indices][:,0]
candidate_fatjets = leading_fatjet

# ### jet matching


d_PDGID = 1
u_PDGID = 2
s_PDGID = 3
b_PDGID = 5
c_PDGID = 4
TOP_PDGID = 6

g_PDGID = 21

ELE_PDGID = 11
vELE_PDGID = 12
MU_PDGID = 13
vMU_PDGID = 14
TAU_PDGID = 15
vTAU_PDGID = 16

Z_PDGID = 23
W_PDGID = 24
H_PDGID = 25

#define deltaR
deltaR = 0.8


#start to compute n_b_tag
def find_closest_SF_awkward(json_path, arr):
    with open(json_path, 'r') as json_file:
        json_SF = json.load(json_file)

    # 提取 pT 和 SF 的值
    pT_values = np.array([item['pT'] for item in json_SF])
    SF_values = np.array([item['SF'] for item in json_SF])
    
    # 创建存放结果的awkward数组
    SF_result = []

    for subarr in arr:
        sub_result = []
        for pT_i in subarr:
            # 查找最接近的 pT 值的索引
            idx = (np.abs(pT_values - pT_i)).argmin()
            # 获取对应的 SF 值
            sub_result.append(SF_values[idx])
        SF_result.append(sub_result)
    
    return ak.Array(SF_result)

pT_distribution = events["JetPUPPI"].PT

mis_tag_udsg = {
    "Tight" : 0.001,
    "Medium": 0.01,
    "Loose" : 0.1,
    "Ideal" : 0,
}

mis_tag_c = {
    "Tight" : 0.04,
    "Medium": 0.13,
    "Loose" : 0.3,
    "Ideal" : 0,
}

json_dir = "./json/28Aug24"
b_tag_eff_loose =  find_closest_SF_awkward(json_path=f"{json_dir}/b_tag_eff_loose.json"  , arr = pT_distribution)
b_tag_eff_medium = find_closest_SF_awkward(json_path=f"{json_dir}/b_tag_eff_medium.json" , arr = pT_distribution)
b_tag_eff_tight =  find_closest_SF_awkward(json_path=f"{json_dir}/b_tag_eff_tight.json"  , arr = pT_distribution)
b_tag_eff_ideal = ak.ones_like(b_tag_eff_tight)

# mistag
mistag_bvsl_loose  = mis_tag_udsg["Loose"]  * ak.ones_like(b_tag_eff_ideal)
mistag_bvsl_medium = mis_tag_udsg["Medium"] * ak.ones_like(b_tag_eff_ideal)
mistag_bvsl_tight  = mis_tag_udsg["Tight"]  * ak.ones_like(b_tag_eff_ideal)
mistag_bvsl_ideal  = mis_tag_udsg["Ideal"]  * ak.ones_like(b_tag_eff_ideal)

mistag_bvsc_loose  = mis_tag_c["Loose"]  * ak.ones_like(b_tag_eff_ideal)
mistag_bvsc_medium = mis_tag_c["Medium"] * ak.ones_like(b_tag_eff_ideal)
mistag_bvsc_tight  = mis_tag_c["Tight"]  * ak.ones_like(b_tag_eff_ideal)
mistag_bvsc_ideal  = mis_tag_c["Ideal"]  * ak.ones_like(b_tag_eff_ideal)

shape = ak.num(pT_distribution)
# 生成一个与输入awkward数组形状相同的、在0-1之间均匀分布的随机数awkward数组
b_tag_eff_random = ak.Array([np.random.uniform(0, 1, length) for length in shape])
mistag_eff_random = ak.Array([np.random.uniform(0, 1, length) for length in shape])

def compare_awkward_arrays(arr, arr2):
    # 逐元素比较 arr2 和 arr，生成布尔awkward数组，条件满足时填1，否则填0
    arr_bool = ak.Array([np.where(a2 < a, 1, 0) for a, a2 in zip(arr, arr2)])
    return arr_bool

b_tag_tight  = compare_awkward_arrays(b_tag_eff_tight,  b_tag_eff_random)
b_tag_medium = compare_awkward_arrays(b_tag_eff_medium, b_tag_eff_random)
b_tag_loose  = compare_awkward_arrays(b_tag_eff_loose,  b_tag_eff_random)
b_tag_ideal  = compare_awkward_arrays(b_tag_eff_ideal,  b_tag_eff_random)

mis_tag_udsg_tight  = compare_awkward_arrays(mistag_bvsl_tight  ,  mistag_eff_random)
mis_tag_udsg_medium = compare_awkward_arrays(mistag_bvsl_medium ,  mistag_eff_random)
mis_tag_udsg_loose  = compare_awkward_arrays(mistag_bvsl_loose  ,  mistag_eff_random)
mis_tag_udsg_ideal  = compare_awkward_arrays(mistag_bvsl_ideal  ,  mistag_eff_random)

mis_tag_c_tight  = compare_awkward_arrays(mistag_bvsc_tight   ,  mistag_eff_random)
mis_tag_c_medium = compare_awkward_arrays(mistag_bvsc_medium  ,  mistag_eff_random)
mis_tag_c_loose  = compare_awkward_arrays(mistag_bvsc_loose   ,  mistag_eff_random)
mis_tag_c_ideal  = compare_awkward_arrays(mistag_bvsc_ideal   ,  mistag_eff_random)


c_PDGID = 4
b_PDGID = 5

b_tag_gen  = ak.values_astype((events["JetPUPPI"].Flavor ==  b_PDGID), int)
c_tag_gen      = ak.values_astype((events["JetPUPPI"].Flavor ==  c_PDGID), int)
light_tag_gen  = ak.values_astype((events["JetPUPPI"].Flavor !=  b_PDGID) & ((events["JetPUPPI"].Flavor !=  c_PDGID)), int)

#clean
dr_AK8_cand_mu = delta_r(candidate_fatjets, files["TTbar_semilep"]["Muon"])
dr_AK8_cand_ele = delta_r(candidate_fatjets, files["TTbar_semilep"]["Electron"])
condition_mu  = ak.all(dr_AK8_cand_mu > 0.8, axis=1) | (ak.is_none(dr_AK8_cand_mu))
condition_ele = ak.all(dr_AK8_cand_ele > 0.8, axis=1) | (ak.is_none(dr_AK8_cand_ele))

# 通过逻辑 OR 运算得到满足条件的索引布尔数组
is_clean_Wcb = condition_mu & condition_ele
# is_clean_Wcb

dr_AK8_cand_AK4_all = delta_r(candidate_fatjets,events["JetPUPPI"])

b_exclusive = ak.values_astype((dr_AK8_cand_AK4_all > 0.8), int)

# b_tag_tight_truth =  events["JetPUPPI"].BTag * b_tag_tight
# b_tag_medium_truth = events["JetPUPPI"].BTag * b_tag_medium
# b_tag_loose_truth =  events["JetPUPPI"].BTag * b_tag_loose

# truth match and correctly tagged
b_tag_tight_truth =  b_exclusive * b_tag_gen * b_tag_tight
b_tag_medium_truth = b_exclusive * b_tag_gen * b_tag_medium
b_tag_loose_truth =  b_exclusive * b_tag_gen * b_tag_loose
b_tag_ideal_truth =  b_exclusive * b_tag_gen * b_tag_ideal

# mis-tag
c_mis_tag_tight_truth  =  b_exclusive * c_tag_gen * mis_tag_c_tight 
c_mis_tag_medium_truth =  b_exclusive * c_tag_gen * mis_tag_c_medium
c_mis_tag_loose_truth  =  b_exclusive * c_tag_gen * mis_tag_c_loose 
c_mis_tag_ideal_truth  =  b_exclusive * c_tag_gen * mis_tag_c_ideal 

light_mis_tag_tight_truth  =  b_exclusive * light_tag_gen * mis_tag_udsg_tight 
light_mis_tag_medium_truth =  b_exclusive * light_tag_gen * mis_tag_udsg_medium
light_mis_tag_loose_truth  =  b_exclusive * light_tag_gen * mis_tag_udsg_loose 
light_mis_tag_ideal_truth  =  b_exclusive * light_tag_gen * mis_tag_udsg_ideal 

tight_tagged_b_jet_idx  = b_tag_tight_truth  + c_mis_tag_tight_truth  + light_mis_tag_tight_truth
medium_tagged_b_jet_idx = b_tag_medium_truth + c_mis_tag_medium_truth + light_mis_tag_medium_truth
loose_tagged_b_jet_idx  = b_tag_loose_truth  + c_mis_tag_loose_truth  + light_mis_tag_loose_truth
ideal_tagged_b_jet_idx  = b_tag_ideal_truth  + c_mis_tag_ideal_truth  + light_mis_tag_ideal_truth

n_b_tight = ak.sum(b_tag_tight_truth, axis = 1)
n_b_medium = ak.sum(b_tag_medium_truth, axis = 1)
n_b_loose = ak.sum(b_tag_loose_truth, axis = 1)
n_b_ideal = ak.sum(b_tag_ideal_truth, axis = 1)

has_two_tight_b =  (n_b_tight == 2)
has_two_medium_b = (n_b_medium == 2)
has_two_loose_b =  (n_b_loose == 2)
has_two_ideal_b =  (n_b_ideal == 2)

def get_two_b_jet(JetArr, tagged_b_jet_idx, has_two_b):
    b_jets = ak.where(tagged_b_jet_idx == 1, JetArr, -99)
    filtered_jets = ak.mask(b_jets, has_two_b)
    is_none = ak.is_none(filtered_jets)
    res = ak.where(is_none, -99, filtered_jets)
    
    def filter_jets(subarray):
    # 如果是数字 0（代表之前是 None），直接返回 0
        if type(subarray) == int:
            return -99
        # 对于非空子数组，提取其中的 JetPUPPI 元素
        jets_only = [item for item in subarray if type(item) != int]
        # 如果找到两个 JetPUPPI 元素，则返回它们，如果少于两个，返回找到的内容
        return jets_only[:2] if len(jets_only) >= 2 else jets_only

    # 3. 使用 map 操作将函数应用于每个子数组
    result = ak.Array([filter_jets(subarray) for subarray in res])

    def extract_first_jet(subarray):
        if type(subarray) == int:  # 如果该位置是 0，则返回 0
            return -99
        else:
            return subarray[0]  # 返回子数组中的第一个 JetPUPPI
    
    def extract_second_jet(subarray):
        if type(subarray) == int:  # 如果该位置是 0，则返回 0
            return -99
        else:
            return subarray[1]  # 返回子数组中的第一个 JetPUPPI

    # 使用列表推导式遍历 Awkward 数组并提取第一个 JetPUPPI 或保留 0
    result_first  = ak.Array([extract_first_jet(subarray) for subarray in result])
    result_second = ak.Array([extract_second_jet(subarray) for subarray in result])

    return result_first, result_second

JetArr = events["JetPUPPI"]

tight_a, tight_b   = get_two_b_jet(JetArr,   tight_tagged_b_jet_idx, has_two_tight_b)
medium_a, medium_b = get_two_b_jet(JetArr, medium_tagged_b_jet_idx, has_two_medium_b)
loose_a, loose_b   = get_two_b_jet(JetArr, loose_tagged_b_jet_idx, has_two_loose_b )
ideal_a, ideal_b   = get_two_b_jet(JetArr, ideal_tagged_b_jet_idx, has_two_ideal_b )

def process_jets(jet1, jet2):
    if type(jet1) == int and type(jet2) == int:  # 如果两个数组的当前位置都是 0
        return -99
    elif type(jet1) != int and type(jet2) != int:  # 如果是 JetPUPPI 对象
        return delta_r(jet1, jet2)  # 返回 Jet1.Eta - Jet2.Eta
    else:
        raise ValueError("Arrays do not match in shape or type.")

def process_jets_between_b1_Wcb(jet1, jet2, Wcb):
    if type(jet1) == int and type(jet2) == int:  # 如果两个数组的当前位置都是 0
        return -99
    elif type(jet1) != int and type(jet2) != int:  # 如果是 JetPUPPI 对象
        return delta_r(jet1, Wcb)  # 返回 Jet1.Eta - Jet2.Eta
    else:
        raise ValueError("Arrays do not match in shape or type.")

def process_jets_between_b2_Wcb(jet1, jet2, Wcb):
    if type(jet1) == int and type(jet2) == int:  # 如果两个数组的当前位置都是 0
        return -99
    elif type(jet1) != int and type(jet2) != int:  # 如果是 JetPUPPI 对象
        return delta_r(jet2, Wcb)  # 返回 Jet1.Eta - Jet2.Eta
    else:
        raise ValueError("Arrays do not match in shape or type.")

def process_jets_deta(jet1, jet2):
    if type(jet1) == int and type(jet2) == int:  # 如果两个数组的当前位置都是 0
        return -99
    elif type(jet1) != int and type(jet2) != int:  # 如果是 JetPUPPI 对象
        return jet1.Eta - jet2.Eta  # 返回 Jet1.Eta - Jet2.Eta
    else:
        raise ValueError("Arrays do not match in shape or type.")

def process_jets_product_eta(jet1, jet2):
    if type(jet1) == int and type(jet2) == int:  # 如果两个数组的当前位置都是 0
        return -99
    elif type(jet1) != int and type(jet2) != int:  # 如果是 JetPUPPI 对象
        return jet1.Eta * jet2.Eta  # 返回 Jet1.Eta - Jet2.Eta
    else:
        raise ValueError("Arrays do not match in shape or type.")
    
delta_r_tight  = ak.Array([process_jets(jet1, jet2) for jet1, jet2 in zip(tight_a, tight_b)])
delta_r_medium = ak.Array([process_jets(jet1, jet2) for jet1, jet2 in zip(medium_a, medium_b)])
delta_r_loose  = ak.Array([process_jets(jet1, jet2) for jet1, jet2 in zip(loose_a, loose_b  )])
delta_r_ideal  = ak.Array([process_jets(jet1, jet2) for jet1, jet2 in zip(ideal_a, ideal_b  )])

delta_r1_Wcb_tight  = ak.Array([process_jets_between_b1_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(tight_a, tight_b  , candidate_fatjets)])
delta_r1_Wcb_medium = ak.Array([process_jets_between_b1_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(medium_a, medium_b, candidate_fatjets)])
delta_r1_Wcb_loose  = ak.Array([process_jets_between_b1_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(loose_a, loose_b  , candidate_fatjets)])
delta_r1_Wcb_ideal  = ak.Array([process_jets_between_b1_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(ideal_a, ideal_b  , candidate_fatjets)])

delta_r2_Wcb_tight  = ak.Array([process_jets_between_b2_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(tight_a, tight_b  , candidate_fatjets)])
delta_r2_Wcb_medium = ak.Array([process_jets_between_b2_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(medium_a, medium_b, candidate_fatjets)])
delta_r2_Wcb_loose  = ak.Array([process_jets_between_b2_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(loose_a, loose_b  , candidate_fatjets)])
delta_r2_Wcb_ideal  = ak.Array([process_jets_between_b2_Wcb(jet1, jet2, Wcb) for jet1, jet2, Wcb in zip(ideal_a, ideal_b  , candidate_fatjets)])

delta_eta_tight       = ak.Array([process_jets_deta(jet1, jet2) for jet1, jet2 in zip(tight_a, tight_b  )])
delta_eta_medium      = ak.Array([process_jets_deta(jet1, jet2) for jet1, jet2 in zip(medium_a, medium_b)])
delta_eta_loose       = ak.Array([process_jets_deta(jet1, jet2) for jet1, jet2 in zip(loose_a, loose_b  )])
delta_eta_ideal       = ak.Array([process_jets_deta(jet1, jet2) for jet1, jet2 in zip(ideal_a, ideal_b  )])

delta_prod_eta_tight  = ak.Array([process_jets_product_eta(jet1, jet2) for jet1, jet2 in zip(tight_a, tight_b  )])
delta_prod_eta_medium = ak.Array([process_jets_product_eta(jet1, jet2) for jet1, jet2 in zip(medium_a, medium_b)])
delta_prod_eta_loose  = ak.Array([process_jets_product_eta(jet1, jet2) for jet1, jet2 in zip(loose_a, loose_b  )])
delta_prod_eta_ideal  = ak.Array([process_jets_product_eta(jet1, jet2) for jet1, jet2 in zip(ideal_a, ideal_b  )])

# maximum and minumum
delta_r_max_b_Wcb_tight   = np.maximum(delta_r1_Wcb_tight  , delta_r2_Wcb_tight )
delta_r_max_b_Wcb_medium  = np.maximum(delta_r1_Wcb_medium , delta_r2_Wcb_medium)
delta_r_max_b_Wcb_loose   = np.maximum(delta_r1_Wcb_loose  , delta_r2_Wcb_loose )
delta_r_max_b_Wcb_ideal   = np.maximum(delta_r1_Wcb_ideal  , delta_r2_Wcb_ideal )

delta_r_min_b_Wcb_tight   = np.minimum(delta_r1_Wcb_tight  , delta_r2_Wcb_tight )
delta_r_min_b_Wcb_medium  = np.minimum(delta_r1_Wcb_medium , delta_r2_Wcb_medium)
delta_r_min_b_Wcb_loose   = np.minimum(delta_r1_Wcb_loose  , delta_r2_Wcb_loose )
delta_r_min_b_Wcb_ideal   = np.minimum(delta_r1_Wcb_ideal  , delta_r2_Wcb_ideal )

# output_file = "/data/bond/zhaoyz/Pheno/slimmedtree/slim_" + delphes_roots["TTbar_semilep"].split("/")[-1] 
output_file = options.outfile

with uproot.recreate(output_file) as root_file:
    root_file["PKUTree"] = {
        "PT_j": np.array(candidate_fatjets.PT),
        "Eta_j": np.array(candidate_fatjets.Eta),
        "Phi_j": np.array(candidate_fatjets.Phi),
        "Mass_j": np.array(candidate_fatjets.Mass),
        "Mass_j_sd": np.array(candidate_fatjets.SoftDroppedP4_5[...,0].mass),
        
        "n_b_tight" : np.array(n_b_tight),
        "n_b_medium" : np.array(n_b_medium),
        "n_b_loose" : np.array(n_b_loose),
        "n_b_ideal" : np.array(n_b_ideal),
        
        "delta_r_tight" : np.array(delta_r_tight),
        "delta_r_medium" : np.array(delta_r_medium),
        "delta_r_loose" : np.array(delta_r_loose),
        "delta_r_ideal" : np.array(delta_r_ideal),      
        
        "delta_r1_Wcb_tight" : np.array(delta_r1_Wcb_tight ),
        "delta_r1_Wcb_medium": np.array(delta_r1_Wcb_medium),
        "delta_r1_Wcb_loose" : np.array(delta_r1_Wcb_loose ),
        "delta_r1_Wcb_ideal" : np.array(delta_r1_Wcb_ideal ),       
              
        "delta_r2_Wcb_tight" : np.array(delta_r2_Wcb_tight ),
        "delta_r2_Wcb_medium": np.array(delta_r2_Wcb_medium),
        "delta_r2_Wcb_loose" : np.array(delta_r2_Wcb_loose ),
        "delta_r2_Wcb_ideal" : np.array(delta_r2_Wcb_ideal ),    
           
        "delta_r_max_b_Wcb_tight"  : delta_r_max_b_Wcb_tight ,
        "delta_r_max_b_Wcb_medium" : delta_r_max_b_Wcb_medium,
        "delta_r_max_b_Wcb_loose"  : delta_r_max_b_Wcb_loose ,
        "delta_r_max_b_Wcb_ideal"  : delta_r_max_b_Wcb_ideal ,

        "delta_r_min_b_Wcb_tight"  : delta_r_min_b_Wcb_tight ,
        "delta_r_min_b_Wcb_medium" : delta_r_min_b_Wcb_medium,
        "delta_r_min_b_Wcb_loose"  : delta_r_min_b_Wcb_loose ,
        "delta_r_min_b_Wcb_ideal"  : delta_r_min_b_Wcb_ideal ,
        
        "delta_eta_tight"  :  np.array(delta_eta_tight),  
        "delta_eta_medium" :  np.array(delta_eta_medium),  
        "delta_eta_loose"  :  np.array(delta_eta_loose),  
        "delta_eta_ideal"  :  np.array(delta_eta_ideal),  

        "delta_prod_eta_tight" : np.array(delta_prod_eta_tight),
        "delta_prod_eta_medium": np.array(delta_prod_eta_medium),
        "delta_prod_eta_loose" : np.array(delta_prod_eta_loose),
        "delta_prod_eta_ideal" : np.array(delta_prod_eta_ideal),  
          
        "is_clean_Wcb" : ak.Array(np.array(is_clean_Wcb).astype(int)),
        "NAK8" : ak.num(events.JetPUPPIAK8.Eta, axis = 1),
        "NAK4" : ak.num(events.JetPUPPI.Eta, axis = 1),
        "hbcvsqcd" : np.array(leading_hbcvsqcd),
        "hbc" : np.array(leading_hbc),
        "qcd" : np.array(leading_qcd),
        "hbq" : np.array(leading_hbq),
    }

print(f"Done transfer tree for {options.infile}")

