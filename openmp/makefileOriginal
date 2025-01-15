# Nome do programa
PROGRAM = benchmark.exe
PROGRAM2 = bench.exe
# Compilador
CC = gcc
# Flags de compilação
CFLAGS = -fopenmp -O3
# Nome do arquivo-fonte
SRC = benchmark.c
SRC2 = bench.c
# Número de threads a serem testados
THREADS = 2 4 8 16
# Número de execuções para cada configuração
RUNS = 5
# Nome do arquivo de saída
OUTPUT_FILE = benchmark.txt
OUTPUT_FILE2 = bench.txt

# Alvo padrão: compilar e executar
all: $(PROGRAM) $(PROGRAM2) run

# Alvo para compilar o programa
$(PROGRAM): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(PROGRAM)

# Alvo para compilar o programa
$(PROGRAM2): $(SRC2)
	$(CC) $(CFLAGS) $(SRC2) -o $(PROGRAM2)


# Alvo para executar o programa com diferentes números de threads e múltiplas execuções
run: $(PROGRAM) $(PROGRAM2)
	@echo "Executando com diferentes números de threads (1 execuções cada):"
	@for t in $(THREADS); do \
		echo "Executando com $$t threads:"; \
		for i in $(shell seq 1 $(RUNS)); do \
			echo "Execução $$i com $$t threads:"; \
			./$(PROGRAM) $$t >> $(OUTPUT_FILE); \
		done; \
		echo ""; \
	done

	@echo "Executando com diferentes números de threads (1 execuções cada):"
	@for t in $(THREADS); do \
		echo "Executando com $$t threads:"; \
		for i in $(shell seq 1 $(RUNS)); do \
			echo "Execução $$i com $$t threads:"; \
			./$(PROGRAM2) $$t >> $(OUTPUT_FILE2); \
		done; \
		echo ""; \
	done
	
# Alvo para limpar arquivos compilados
clean:
	rm -f $(PROGRAM) $(PROGRAM2)

