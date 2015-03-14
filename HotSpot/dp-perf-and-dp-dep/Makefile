#Makefile
# Created on: Jan 28, 2015
#     Author: Jie Shen
# Revised on: Feb 8, 2015 

CC = mcc
CXX = mcxx
CCFLAGS = -g -O3 -Wall -Wno-unused #-k

ICC = icc
ICCFLAGS = -g -O3 -Wall -Wno-unused -vec-report


MCXX_FLAGS   = --ompss
MCXX_FLAGS_P =
MCXX_FLAGS_I = --instrument
MCXX_FLAGS_D = --debug

KERNEL=kernel.cl
PREFIX=.

LIB = -lm -lOpenCL
INC  = -I$(PREFIX) --opencl-build-opts=-I$(PREFIX)

ICCLIB = -lsvml

PRG = hotspot
EXE_P = $(PRG)-p
EXE_I = $(PRG)-i
EXE_D = $(PRG)-d
EXEALL = $(EXE_P) $(EXE_I) $(EXE_D)
 
SRC = hotspot.c
OBJ = jutil.o golden.o
SRC_SMP_VEC = hotspot-smp-vec.c
OBJ_SMP_VEC = hotspot-smp-vec.o

all: $(EXEALL)

$(EXE_P): $(SRC) $(OBJ) $(OBJ_SMP_VEC) $(KERNEL) 
	$(CC) $(CCFLAGS) $(MCXX_FLAGS) $(MCXX_FLAGS_P) $(INC) $(LIB) $(ICCLIB) $(SRC) $(OBJ) $(OBJ_SMP_VEC) $(KERNEL) -o $(EXE_P)
	
$(EXE_I): $(SRC) $(OBJ) $(OBJ_SMP_VEC) $(KERNEL) 
	$(CC) $(CCFLAGS) $(MCXX_FLAGS) $(MCXX_FLAGS_I) $(INC) $(LIB) $(ICCLIB) $(SRC) $(OBJ) $(OBJ_SMP_VEC) $(KERNEL) -o $(EXE_I)
	
$(EXE_D): $(SRC) $(OBJ) $(OBJ_SMP_VEC) $(KERNEL) 
	$(CC) $(CCFLAGS) $(MCXX_FLAGS) $(MCXX_FLAGS_D) $(INC) $(LIB) $(ICCLIB) $(SRC) $(OBJ) $(OBJ_SMP_VEC) $(KERNEL) -o $(EXE_D)
	
$(OBJ_SMP_VEC): $(SRC_SMP_VEC)
	$(ICC) $(ICCFLAGS) -c $(SRC_SMP_VEC)

$(OBJ): %.o: %.c
	$(CC) $(CCFLAGS) $< -c

clean:
	rm *.o $(EXEALL)

