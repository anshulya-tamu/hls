for index in 103 302 308 309 324 328 341 351 359 367 385 402 405 412
do

    if test -f "results/${index}".csv; then
        echo "File for ${index} exists."
        rm -r "${index}"
    fi
done