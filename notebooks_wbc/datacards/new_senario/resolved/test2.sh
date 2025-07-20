for i in {4..7}
do
    echo "now lumi: ${i}"
    for j in {0..24}
    do
        cd strength_point_${i}_${j} 
        source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/new_senario/boosted/cmd.sh
        cd -
    done
done
