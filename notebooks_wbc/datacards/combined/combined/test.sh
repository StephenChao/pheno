for i in {0..19}
do
    echo "now ${i}"
    cd strength_1_no_${i} 
    source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/combined_updated/combined/cmd.sh
    cd -

    cd strength_0p5_no_${i} 
    source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/combined_updated/combined/cmd.sh
    cd -

    # cd -
done
