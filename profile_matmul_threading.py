import os
import subprocess

# Configuration
SIZES = [1024]  # Array of matrix sizes to test
THREAD_COUNTS = [2, 4, 6, 8]  # Array of number of threads to test
OUTPUT_DIR = "profiling_results"  # Directory to store profiling results
PMU_SCRIPT = "/classes/ece5755/pmu-tools/toplev.py"  # Path to the PMU script
TEST_PROGRAM = "./test_matmul_threading"  # Name of the compiled test program
MAIN_SOURCE = "./profiling/test_matmul_threading.c"  # Source files for compilation
LIBRARY_SOURCE = "./kernel/*.c"  # Library source

# NUMA node to bind
NUMA_NODE = 0

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compile_program():
    """
    Compiles the test program from the given source files.
    """
    print("Compiling the test program...")
    compile_command = f"gcc -O3 -o {TEST_PROGRAM} {MAIN_SOURCE} {LIBRARY_SOURCE} -lm -lpthread"
    result = subprocess.run(compile_command, shell=True)
    if result.returncode != 0:
        print("Compilation failed. Exiting.")
        exit(1)
    print("Compilation successful.")


def run_baseline_profile(size):
    """
    Runs profiling for single-threaded matrix multiplication with NUMA node binding.
    """
    output_file_txt = f"{OUTPUT_DIR}/baseline_size{size}.txt"
    output_file_csv = f"{OUTPUT_DIR}/baseline_size{size}.csv"
    print(f"Profiling baseline matmul - size: {size}x{size}")
    
    # Run baseline matmul with NUMA binding
    taskset_command = f"numactl --cpunodebind={NUMA_NODE} taskset -c 0 {TEST_PROGRAM} {size} {size} 1 > {output_file_txt} 2>&1"
    subprocess.run(taskset_command, shell=True)

    # Run PMU profiling with NUMA binding
    pmu_command = (
        f"numactl --cpunodebind={NUMA_NODE} python3 {PMU_SCRIPT} --core S0-C0 -l1 -v --no-desc --force-cpu spr "
        f"--csv , --output {output_file_csv} {TEST_PROGRAM} {size} {size} 1"
    )
    subprocess.run(pmu_command, shell=True)


def run_threaded_profile(size, threads):
    """
    Runs profiling for multi-threaded matrix multiplication with NUMA node binding.
    """
    output_file_txt = f"{OUTPUT_DIR}/threaded_size{size}_threads{threads}.txt"
    output_file_csv = f"{OUTPUT_DIR}/threaded_size{size}_threads{threads}.csv"
    core_list = ",".join(map(str, range(0, threads * 2, 2)))  # Adjust core list for taskset
    core_mask = ",".join([f"S0-C{core}" for core in range(0, threads * 2, 2)])  # Adjust core mask for toplev.py
    print(f"Profiling threaded matmul - size: {size}x{size}, threads: {threads}")
    
    # Run threaded matmul with NUMA binding
    taskset_command = f"numactl --cpunodebind={NUMA_NODE} taskset -c {core_list} {TEST_PROGRAM} {size} {size} {threads} > {output_file_txt} 2>&1"
    subprocess.run(taskset_command, shell=True)

    # Run PMU profiling with NUMA binding
    pmu_command = (
        f"numactl --cpunodebind={NUMA_NODE} python3 {PMU_SCRIPT} --core {core_mask} -l1 -v --no-desc --force-cpu spr "
        f"--csv , --output {output_file_csv} {TEST_PROGRAM} {size} {size} {threads}"
    )
    subprocess.run(pmu_command, shell=True)


def main():
    """
    Main function to compile the program and run profiling for all configurations.
    """
    compile_program()

    for size in SIZES:
        # Uncomment the following line to enable baseline profiling
        run_baseline_profile(size)
        for threads in THREAD_COUNTS:
            run_threaded_profile(size, threads)

    print(f"Profiling completed. Results are in the {OUTPUT_DIR} directory.")


if __name__ == "__main__":
    main()
