### ihep sample production documentation

This file is an instruction for Delphes sample production on ihep server. 

```
# firstly load the environment

source /cvmfs/cms.cern.ch/cmsset_default.sh
source /scratchfs/cms/licq/utils/load_standalonemg_env.sh
export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin/:$PATH

rsync -r /publicfs/cms/user/licq/pheno/anomdet/gen/condor/glopart_training_2312 . --exclude="logs*"
cd glopart_training_2312


# test the production method locally
./run_sm.sh sm/TTbar 1000 1000 0 8888 ihep

```

If want to use Condor to product the samples, refer to the [note](https://github.com/colizz/anomdet_pheno/blob/master/condor/ihep.md) and the official [link](http://afsapply.ihep.ac.cn/cchelp/en/local-cluster/jobs/HTCondor/#3211-introduction).

The example to submit Condor jobs:

```
hep_sub run_sm.sh -argu sm/TTbar 10000 5000 "%{ProcId}" 0 ihep -n 200

# here the parameters mean: 10000 events per job, one job will further split to minijob/batch in the Delphes production module, every mini job will run 5000 events, so there will be 10000/5000=2 batchs here. Then 200 jobs will be submitted.

# If using slc7, using commands like:

hep_sub -os CentOS7 run_sm.sh -argu sm/TTbar_semilep 40000 5000 "%{ProcId}" 0 ihep -n 300
```

After the Delphes root production, using the following command example to run the ntuple (e.g.,see pheno/production/ntuplizer/sophon/analyzers/run_local.sh):

```
root -b -q 'infer_sophon.C++("events_delphes_2.root", "out.root", "JetClassII_Sophon.onnx")'
```


# MG + Pythia + Delphes Environment & Sample Generation Notes

## 1. Software Environment (if you want to perform a fresh manual setup)

- Based on `CMSSW_11_1_0_pre5_PY3`
- Download `MG5_aMC_v2_9_18` and inside it install the required packages:
  - `install pythia8`
  - `install lhapdf6_py3`
  - `...` (other required dependencies)
- Install `Delphes` normally

---

## 2. Specifying / Generating Samples

- The `genpacks` directory contains all **MG + Pythia** cards.
  - In MadGraph **step2** you can specify parameters / variables, e.g. the example `tagger/train_hbb`.
  - The Pythia card is standard (note that configurations differ under different **MLM matching** setups, e.g. compare `sm/TTbar` vs `sm/TZ`).
  - The Pythia card can also define custom decay modes, e.g. `sm_ext/SingleHiggsToBB`.
- `Delphes` has been customized (patched) to `DelphesHepMC2WithFilter`.
  - The file was modified:  
    original: `/scratchcfs/cms/licq/utils/Delphes-3.5.0/DelphesHepMC2.cpp`  
    renamed / adapted to: `DelphesHepMC2WithFilter.cpp`

---
