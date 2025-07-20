# text2workspace.py datacard.txt 
# # combine -M AsymptoticLimits -d datacard.root --expectSignal 1 -t -1 --run expected 2>&1 | tee ./ExpectedAsymptoticLimits.txt

# combine -M AsymptoticLimits -d datacard.root 2>&1 | tee ./ObservedAsymptoticLimits.txt

# # combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. 2>&1 | tee ./TotalUnc.txt

# # combine -M Significance -d datacard.root 2>&1 | tee ./ObservedSignificance.txt

for i in $(seq 60 10 160); do
    echo $i
    cd sig_m${i}
    source /home/pku/zhaoyz/pheno/notebooks_hbc/datacards/cmd.sh
    cd -
done