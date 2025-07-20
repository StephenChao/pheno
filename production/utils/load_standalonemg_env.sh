cd /scratchfs/cms/licq/utils/CMSSW_11_1_0_pre5_PY3/src
cmsenv
cd -
export PYTHIA8DATA=/scratchfs/cms/licq/utils/MG5_aMC_v2_9_18/HEPTools/pythia8/share/Pythia8/xmldoc
export SRT_PYTHIA8DATA_SCRAMRT=$PYTHIA8DATA
export SRT_PYTHIA8DATA_SCRAMRTDEL=$PYTHIA8DATA

# export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/data/pku/home/licq/utils/MG5_aMC_v2_7_3/Delphes/external
# export SRT_ROOT_INCLUDE_PATH_SCRAMRTDEL=$SRT_ROOT_INCLUDE_PATH_SCRAMRTDEL:/data/pku/home/licq/utils/MG5_aMC_v2_7_3/Delphes/external
# now use Delphes3.5.0
export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/scratchfs/cms/licq/utils/Delphes-3.5.0/external
export SRT_ROOT_INCLUDE_PATH_SCRAMRTDEL=$SRT_ROOT_INCLUDE_PATH_SCRAMRTDEL:/scratchfs/cms/licq/utils/Delphes-3.5.0/external

# LHAPDF
# LHAPDFCONFIG=/scratchfs/cms/licq/utils/lhapdf/lhapdf-install-py2/bin/lhapdf-config
LHAPDFCONFIG=/scratchfs/cms/licq/utils/MG5_aMC_v2_9_18/HEPTools/lhapdf6_py3/bin/lhapdf-config
LHAPDFINCLUDES=`$LHAPDFCONFIG --incdir`
LHAPDFLIBS=`$LHAPDFCONFIG --libdir`

export LHAPDF_DATA_PATH=/scratchfs/cms/licq/utils/MG5_aMC_v2_9_18/HEPTools/lhapdf6_py3/share/LHAPDF

