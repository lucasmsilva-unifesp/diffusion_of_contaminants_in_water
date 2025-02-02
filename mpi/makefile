# Compiler and flags
CC = mpicc
CFLAGS = -Wall -O3
NPROCS ?= 4

# Source files
SRCS = benchmarch_linear.c mpi_2d.c
EXES = $(SRCS:.c=)

# Output files
BENCH_FILE = benchmarch_linear_times.txt
MPI_FILE = mpi_2d_times.txt

# Default target
all: $(EXES)

# Compile each source file into an executable
%: %.c
	$(CC) $(CFLAGS) $< -o $@

# Run each executable and store execution times in separate files
run: $(EXES)
	@echo "Running tests and saving execution times..."
	@echo "" >> $(BENCH_FILE)
	@echo "" >> $(MPI_FILE)

	@for exe in $(EXES); do \
		if [ "$$exe" = "benchmarch_linear" ]; then \
			OUT_FILE=$(BENCH_FILE); \
		else \
			OUT_FILE=$(MPI_FILE); \
		fi; \
		for np in $$(seq 1 8); do \
			echo "Running $$exe with $$np processes..." | tee -a $$OUT_FILE; \
			for i in 1 2 3; do \
				echo -n "$$exe;$$np;" >> $$OUT_FILE; \
				mpirun -np $$np ./$$exe | grep "tempo de execução" | awk '{print $$4}' >> $$OUT_FILE; \
			done; \
		done; \
	done

clean:
	rm -f $(EXES)

.PHONY: all run clean
