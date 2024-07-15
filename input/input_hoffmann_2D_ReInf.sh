
for filename in roughness/*.json; do
    echo $filename

    mpirun -np 20 ./gls-app $filename | tee ${filename::-4}"txt"
    cp $(ls -t *.vtu | head -1) ${filename::-4}"vtu"
done
