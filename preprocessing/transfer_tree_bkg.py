import os
import subprocess
from optparse import OptionParser
from JobManager import concurrent_jobs, submit, wait


parser = OptionParser()
parser.add_option('--indir', help='input directory for delphes root file')
parser.add_option('--outdir', help='input directory for output file')
parser.add_option('--test', action="store_true", help='request a dry run')
parser.add_option('--jobs')
(options, args) = parser.parse_args()
if options.jobs: concurrent_jobs(int(options.jobs))

count = 50
i = 0
for file in os.listdir(options.indir):
    #naive check:
    i += 1
    # if i == count: break
    if file.startswith("ntuple"):
        # cmd = f"stat {options.outdir}/slim_{file} &>/dev/null || (python gen_matching.py --infile {options.indir}/{file} --outfile {options.outdir}/slim_{file} &> outs/{file}.log)"
        cmd = f"stat {options.outdir}/slim_{file} &>/dev/null || (python btag_bkg.py --infile {options.indir}/{file} --outfile {options.outdir}/slim_{file} &> outs/{file}.log)"
        # print(f"do:{cmd}")
        if options.test:
            print(f"do:{cmd}")
        else:
            submit(cmd)
            if getattr(options, 'jobs') == '1': wait()
wait()

#typical command:
#python transfer_tree.py  --indir /data/bond/zhaoyz/Pheno/ntuplizer/output/sm/TTbar_semilep_set1_12M --outdir /data/bond/zhaoyz/Pheno/slimmedtree/sm/TTbar_semilep_set1_12M --jobs 100