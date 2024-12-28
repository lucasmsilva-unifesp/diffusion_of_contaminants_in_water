# Nome do programa
PROGRAM = benchmark.exe
# Compilador
CC = gcc
# Flags de compilação
CFLAGS = -fopenmp -O3
# Nome do arquivo-fonte
SRC = benchmark.c
# Número de threads a serem testados
THREADS = 2 4 8 16
# Número de execuções para cada configuração
RUNS = 10
# Nome do arquivo de saída
OUTPUT_FILE = c-schedule_benchmark.txt

# Alvo padrão: compilar e executar
all: $(PROGRAM) run

# Alvo para compilar o programa
$(PROGRAM): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(PROGRAM)

# Alvo para executar o programa com diferentes números de threads e múltiplas execuções
run: $(PROGRAM)
	@echo "Executando com diferentes números de threads:"
	@for t in $(THREADS); do \
		echo "Executando com $$t threads:"; \
		for i in $(shell seq 1 $(RUNS)); do \
			echo "\tExecução de número $$i com $$t threads:"; \
			./$(PROGRAM) $$t >> $(OUTPUT_FILE); \
		done; \
		echo ""; \
	done
	
# Alvo para limpar arquivos compilados
clean:
	rm -f $(PROGRAM) $(OUTPUT_FILE)

