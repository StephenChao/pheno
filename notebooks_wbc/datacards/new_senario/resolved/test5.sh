for i in {16..19}
do
    echo "now lumi: ${i}"
    for j in {0..24}
    do
        echo "*************************************************"
        cd strength_point_${i}_${j} 
        echo "strength ${j}"
        source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/new_senario/boosted/cmd.sh
        echo "*************************************************"
        cd -
    done
done
