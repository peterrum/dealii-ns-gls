foldername=roughness_3

for full_name in $foldername/*.json; do
    filename=$(basename ${full_name})
    echo $filename

    mpirun -np 24 ./gls-app $foldername/$filename | tee $foldername/${filename::-4}"txt"
    cp $(ls -t results_hoffmann_2D_ReInf.*.*.vtu | head -1) $foldername/${filename::-4}"vtu"
    cp $(ls -t results_hoffmann_2D_ReInf.*_slice_0_*.vtu | head -1) $foldername/slice_0_${filename::-4}"vtu"
    cp $(ls -t results_hoffmann_2D_ReInf.*_slice_1_*.vtu | head -1) $foldername/slice_1_${filename::-4}"vtu"
    cp $(ls -t results_hoffmann_2D_ReInf.*_drag_lift_pressure.m | head -1) $foldername/${filename::-4}"m"
done
