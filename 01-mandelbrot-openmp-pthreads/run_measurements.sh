#!/bin/bash

MEASUREMENTS=10
INITIAL_SIZE=16
ITERATIONS=10
THREAD_COUNTS=(1 2 4 8 16 32)
CSV_FILE="measurements.csv"

make

echo "version,threads,region,image_size,io,mean_time_sec,std_dev_sec" > "$CSV_FILE"

REGIONS=(
    "full -2.5 1.5 -2.0 2.0"
    "seahorse -0.8 -0.7 0.05 0.15"
    "elephant 0.175 0.375 -0.1 0.1"
    "triple_spiral -0.188 -0.012 0.554 0.754"
)

run_benchmark() {
    local version=$1
    local threads=$2
    local region_name=$3
    local coords=$4
    local size=$5
    local io_flag=$6
    local cmd=$7

    echo "Messung: $version | Threads: $threads | Region: $region_name | Groesse: ${size}x${size} | I/O: $io_flag"

    local output
    output=$(LC_ALL=C perf stat -r "$MEASUREMENTS" bash -c "$cmd" 2>&1)
    
    local mean
    local std_dev
    mean=$(echo "$output" | grep "seconds time elapsed" | awk '{print $1}')
    std_dev=$(echo "$output" | grep "seconds time elapsed" | awk '{print $3}')

    echo "$version,$threads,$region_name,$size,$io_flag,$mean,$std_dev" >> "$CSV_FILE"
}

# 1. Sequenziell: MIT und OHNE I/O
for io_flag in 0 1; do
    for entry in "${REGIONS[@]}"; do
        reg_name=$(echo "$entry" | awk '{print $1}')
        coords=$(echo "$entry" | cut -d' ' -f2-)
        size=$INITIAL_SIZE
        for ((i=1; i<=ITERATIONS; i++)); do
            run_benchmark "seq" 1 "$reg_name" "$coords" "$size" "$io_flag" "./mandelbrot_seq $coords $size $io_flag"
            rm -f output.ppm
            size=$((size * 2))
        done
    done
done

# 2. Pthreads: OHNE I/O
for threads in "${THREAD_COUNTS[@]}"; do
    for entry in "${REGIONS[@]}"; do
        reg_name=$(echo "$entry" | awk '{print $1}')
        coords=$(echo "$entry" | cut -d' ' -f2-)
        size=$INITIAL_SIZE
        for ((i=1; i<=ITERATIONS; i++)); do
            run_benchmark "pth" "$threads" "$reg_name" "$coords" "$size" 1 "./mandelbrot_pth $coords $size $threads 1"
            size=$((size * 2))
        done
    done
done

# 3. OpenMP: OHNE I/O
for threads in "${THREAD_COUNTS[@]}"; do
    for entry in "${REGIONS[@]}"; do
        reg_name=$(echo "$entry" | awk '{print $1}')
        coords=$(echo "$entry" | cut -d' ' -f2-)
        size=$INITIAL_SIZE
        for ((i=1; i<=ITERATIONS; i++)); do
            run_benchmark "omp" "$threads" "$reg_name" "$coords" "$size" 1 "OMP_NUM_THREADS=$threads ./mandelbrot_omp $coords $size 1"
            size=$((size * 2))
        done
    done
done

echo "Fertig! Alle Ergebnisse liegen in $CSV_FILE"