# Nome do programa
PROGRAM = benchmark.exe
# Compilador
CC = gcc
# Flags de compilação
CFLAGS = -O3
# Nome do arquivo-fonte
SRC = benchmark.c
# Número de execuções para cada configuração
RUNS = 5
# Nome do arquivo de saída
OUTPUT_FILE = benchmark.txt

# Alvo padrão: compilar e executar
all: $(PROGRAM) run

# Alvo para compilar o programa
$(PROGRAM): $(SRC)
	$(CC) $(SRC) $(CFLAGS) -o $(PROGRAM)

# Alvo para executar o programa com diferentes números de threads e múltiplas execuções
run: $(PROGRAM)
	@echo "Executando com diferentes números de threads (5 execuções cada):"
	@for i in $(shell seq 1 $(RUNS)); do \
		echo "Execução $$i sequencial:"; \
		./$(PROGRAM) $$t >> $(OUTPUT_FILE); \
	done; \
	@echo ""; \

# Alvo para limpar arquivos compilados
clean:
	rm -f $(PROGRAM) $(OUTPUT_FILE)
