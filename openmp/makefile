CC = gcc
CFLAGS = -O2 -fopenmp
EXEC = testandoDesempenho
SRC = testandoDesempenho.c

all: $(EXEC)

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC)

clean:
	rm -f $(EXEC)

run: $(EXEC)
	./$(EXEC) 4
