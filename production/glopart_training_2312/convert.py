import os
inpath = "/publicfs/cms/user/zhaoyz/root/MiniAOD/TTToSeiLeptonic"
outpath = "/publicfs/cms/user/zhaoyz/root/Delphes/TTToSeiLeptonic"
delphes_path = "/publicfs/cms/user/zhaoyz/utils/Delphes-3.5.0"
i = 0
for file_i in os.listdir(inpath):
    i += 1
    print("running:",file_i)
    os.system(f"{delphes_path}/DelphesCMSFWLite {delphes_path}/cards/converter_card.tcl {outpath}/out_{str(i)}.root {inpath}/{file_i}")    
    
