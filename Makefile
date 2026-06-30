CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -lm
SRCDIR  = program

# Nomes-base de cada algoritmo (sem path e sem .c)
ALGOS = bottomupsort combsort insertionsort mergesort quicksort smoothsort shellsort heapsort countingsort radixsort bitonicsort oddevensort

# Executáveis serial e paralelo para cada algoritmo
SERIAL   = $(addsuffix _serial,$(ALGOS))
PARALLEL = $(addsuffix _parallel,$(ALGOS))

all: $(SERIAL) $(PARALLEL)

# Versão serial: sem a macro PARALLEL definida
%_serial: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $< -o $@

# Versão paralela: define a macro PARALLEL
%_parallel: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -DPARALLEL $< -o $@

clean:
	rm -f $(SERIAL) $(PARALLEL)

.PHONY: all clean
