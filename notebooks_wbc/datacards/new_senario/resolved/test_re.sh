for i in {0..5}
do
    echo "now lumi: ${i}"
    for j in {0..24}
    do
        cd strength_point_${i}_${j} 
        source /home/pku/zhaoyz/pheno/notebooks_v2/datacards/new_senario/resolved/cmd.sh
        cd -
    done
done
