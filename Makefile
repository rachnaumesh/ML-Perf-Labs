CC = gcc
CFLAGS = -I. -lm

HDF5_FLAGS = -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
COMMON_HDRS = ./utils/data_utils.h ./kernel/conv.h ./kernel/matrix_ops.h ./kernel/linear.h ./kernel/functional.h ./kernel/nn.h ./kernel/attention.h
COMMON_SRC = ./utils/data_utils.c ./kernel/conv.c ./kernel/functional.c ./kernel/matrix_ops.c ./kernel/linear.c ./kernel/nn.c ./kernel/attention.c

# Unity test framework
UNITY_FILES = ./tests/unity/unity.c
TEST_FILES = $(wildcard ./tests/*.c)
TEST_EXECUTABLES = $(patsubst %.c,%,$(TEST_FILES))

# Performance
MATMUL_TARGETS = matmul_naive matmul_blocking

BINS = lab2

.PHONY: all
all: $(BINS)

.PHONY: lab2
lab2: lab2.c $(COMMON_HDRS)
	$(CC) -o $@ lab2.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)

.PHONY: test
test: all_tests

.PHONY: all_tests
all_tests: 
	$(CC) -o tests/$@ $(TEST_FILES) $(UNITY_FILES) $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS) 
	./tests/$@

.PHONY: grading
grading:
	bash grading/grading.sh

.PHONY: grading_tests
grading_tests:
	$(CC) -o grading/$@ $(GRADING_FILES) $(UNITY_FILES) $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)
	./grading/$@

.PHONY : $(MATMUL_TARGETS)
$(MATMUL_TARGETS):
	$(CC) -o $@ ./perf/$@.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)
	/usr/local/pmu-tools/pmu-tools/toplev.py --core S0-C0 -l1 -v --no-desc taskset -c 0 ./$@

.PHONY: clean
clean:
	rm -f $(BINS) $(TEST_EXECUTABLES) $(GRADING_EXECUTABLES) $(GRADING_TESTS_OUTPUT) $(MATMUL_TARGETS)