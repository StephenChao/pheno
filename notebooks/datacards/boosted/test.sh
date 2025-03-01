for i in {0..19}
do
    # mkdir no_${i}
    # mv input_SR_no_${i}.root no_${i}
    cd no_${i}
    echo doing ${i}
    # mv * input_SR.root
    source /home/pku/zhaoyz/pheno/notebooks/datacards/cmd.sh 
    cd -
done
