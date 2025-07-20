# note: use special physics model!! Scale bkg by (1 - 0.0008398 * r) / (1 - 0.0008398)

# text2workspace.py datacard.txt -P HiggsAnalysis.CombinedLimit.PhysicsModelCustom:bkgScaling; \
# combine -M Significance datacard.root -t -1 --expectSignal=1; \
# combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5.

# # only best fit
# text2workspace.py datacard.txt -P HiggsAnalysis.CombinedLimit.PhysicsModelCustom:bkgScaling; \
# combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. 2>&1 | tee ./TotalUnc.txt

text2workspace.py datacard.txt 
combine -M AsymptoticLimits -n "" -d datacard.root 2>&1 | tee ./ObservedAsymptoticLimits.txt


# ## significance with stat only
# text2workspace.py datacard.txt -P HiggsAnalysis.CombinedLimit.PhysicsModelCustom:bkgScaling; \
# combine -M Significance datacard.root --freezeParameters var{ftag_.*},wcb_sf -t -1 --expectSignal=1; \
# combine -M MultiDimFit datacard.root --freezeParameters var{ftag_.*},wcb_sf --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5.

# # unc breakdown (split ftag to b/c/l jets)
# text2workspace.py datacard.txt -P HiggsAnalysis.CombinedLimit.PhysicsModelCustom:bkgScaling; \
# combine -M MultiDimFit datacard.root --algo=grid --robustFit=1 --points=200 --cminDefaultMinimizerTolerance 5. -n Grid > /dev/null; \
# combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. -n Bestfit --saveWorkspace > /dev/null; \
# combine -M MultiDimFit --algo=grid --points=200 --cminDefaultMinimizerTolerance 5. higgsCombineBestfit.MultiDimFit.mH120.root --snapshotName MultiDimFit --freezeParameters var{ftag_bjet.*} -n Freeze_till_ftag_bjet > /dev/null; \
# combine -M MultiDimFit --algo=grid --points=200 --cminDefaultMinimizerTolerance 5. higgsCombineBestfit.MultiDimFit.mH120.root --snapshotName MultiDimFit --freezeParameters var{ftag_bjet.*},var{ftag_cjet.*} -n Freeze_till_ftag_cjet > /dev/null; \
# combine -M MultiDimFit --algo=grid --points=200 --cminDefaultMinimizerTolerance 5. higgsCombineBestfit.MultiDimFit.mH120.root --snapshotName MultiDimFit --freezeParameters var{ftag_bjet.*},var{ftag_cjet.*},var{ftag_ljet.*} -n Freeze_till_ftag_ljet > /dev/null; \
# plot1DScanWithOutput.py higgsCombineGrid.MultiDimFit.mH120.root --others higgsCombineFreeze_till_ftag_bjet.MultiDimFit.mH120.root:Freeze_till_ftag_bjet:2 higgsCombineFreeze_till_ftag_cjet.MultiDimFit.mH120.root:Freeze_till_ftag_cjet:3 higgsCombineFreeze_till_ftag_ljet.MultiDimFit.mH120.root:Freeze_till_ftag_ljet:4 -o full_unce_breakdown --breakdown ftag_bjet,ftag_cjet,ftag_ljet,stats

# # unc breakdown
# text2workspace.py datacard.txt -P HiggsAnalysis.CombinedLimit.PhysicsModelCustom:bkgScaling; \
# combine -M MultiDimFit datacard.root --algo=grid --robustFit=1 --points=200 --cminDefaultMinimizerTolerance 5. -n Grid > /dev/null; \
# combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. -n Bestfit --saveWorkspace > /dev/null; \
# combine -M MultiDimFit --algo=grid --points=200 --cminDefaultMinimizerTolerance 5. higgsCombineBestfit.MultiDimFit.mH120.root --snapshotName MultiDimFit --freezeParameters var{ftag_.*} -n Freeze_till_ftag > /dev/null; \
# plot1DScanWithOutput.py higgsCombineGrid.MultiDimFit.mH120.root --others higgsCombineFreeze_till_ftag.MultiDimFit.mH120.root:Freeze_till_ftag:2 -o full_unce_breakdown --breakdown ftag,stats

# unc breakdown for boosted (with wcb_sf)
# text2workspace.py datacard.txt -o datacard.root -P HiggsAnalysis.CombinedLimit.PhysicsModelCustom:bkgScaling; \
# combine -M MultiDimFit datacard.root --algo=grid --robustFit=1 --points=200 --cminDefaultMinimizerTolerance 5. -n Grid > /dev/null; \
# combine -M MultiDimFit datacard.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. -n Bestfit --saveWorkspace > /dev/null; \
# combine -M MultiDimFit --algo=grid --points=200 --cminDefaultMinimizerTolerance 5. higgsCombineBestfit.MultiDimFit.mH120.root --snapshotName MultiDimFit --freezeParameters var{ftag_.*} -n Freeze_till_ftag > /dev/null; \
# combine -M MultiDimFit --algo=grid --points=200 --cminDefaultMinimizerTolerance 5. higgsCombineBestfit.MultiDimFit.mH120.root --snapshotName MultiDimFit --freezeParameters var{ftag_.*},wcb_sf -n Freeze_till_wcb_sf > /dev/null; \
# # plot1DScanWithOutput.py higgsCombineGrid.MultiDimFit.mH120.root --others higgsCombineFreeze_till_ftag.MultiDimFit.mH120.root:Freeze_till_ftag:2 higgsCombineFreeze_till_wcb_sf.MultiDimFit.mH120.root:Freeze_till_wcb_sf:3 -o full_unce_breakdown --breakdown ftag,wcb_sf,stats
# python3 /home/pku/zhaoyz/backup/CMSSW_11_3_4/src/CombineHarvester/CombineTools/scripts/plot1DScanWithOutput.py higgsCombineGrid.MultiDimFit.mH120.root --others higgsCombineFreeze_till_ftag.MultiDimFit.mH120.root:Freeze_till_ftag:2 higgsCombineFreeze_till_wcb_sf.MultiDimFit.mH120.root:Freeze_till_wcb_sf:3 -o full_unce_breakdown --breakdown ftag,wcb_sf,stats 2>&1 | tee ./BreakDown.txt

# # combine cards
# combineCards.py ../../boosted/datacard.txt ../../resolved/datacard.txt > datacard.txt