text2workspace.py datacard.txt 
# combine -M AsymptoticLimits -d datacard.root --expectSignal 1 -t -1 --run expected 2>&1 | tee ./ExpectedAsymptoticLimits.txt

combine -M AsymptoticLimits -d datacard.root 2>&1 | tee ./ObservedAsymptoticLimits.txt

# combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. 2>&1 | tee ./TotalUnc.txt

# combine -M Significance -d datacard.root 2>&1 | tee ./ObservedSignificance.txt
