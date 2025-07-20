for i in {0..19}
do
    # mkdir no_${i}
    mkdir strength_0p5_no_${i}
    mkdir strength_1_no_${i}
    # mv no_${i} strength_1_no_${i}
    # mv input_SR_no_${i}.root no_${i}
    # cd no_${i}
    # echo doing ${i}
    # # mv * input_SR.root
    # source /home/pku/zhaoyz/pheno/notebooks/datacards/cmd.sh 

    # cd strength_0p5_no_${i} 
    # cp ../../boosted/strength_0p5_no_${i}/*SR*.root .
    # cp ../../resolved/strength_0p5_no_${i}/*SR*.root .
    # combineCards.py ../../boosted/strength_0p5_no_${i}/datacard.txt ../../resolved/strength_0p5_no_${i}/datacard.txt > datacard.txt
    # cd -

    # cd strength_1_no_${i} 
    # cp ../../boosted/strength_1_no_${i}/*SR*.root .
    # cp ../../resolved/strength_1_no_${i}/*SR*.root .
    # combineCards.py ../../boosted/strength_1_no_${i}/datacard.txt ../../resolved/strength_1_no_${i}/datacard.txt > datacard.txt
    # cd -

    # cd strength_1_no_${i} 
    # source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/cmd.sh
    # cd -

    # cd strength_0p5_no_${i} 
    # source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/cmd.sh
    # cd -
    # cd -
done
