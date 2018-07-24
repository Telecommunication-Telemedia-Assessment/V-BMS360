# the SOURCE definiton lets move your makefile to another position
CONFIG 			= LIBRARY

# set pathes to the correct directories
SRC_DIR			= ./src
INC_DIR			= 
LIB_DIR			= ./lib
BIN_DIR			= 

SRC_DIR1		=
SRC_DIR2		=
SRC_DIR3		=
SRC_DIR4		=


USER_INC_DIRS	= -I$(SRC_DIR) \
				  -I/usr/local/opt/opencv/include

USER_LIB_DIRS	= 



# intermediate directory for object files
OBJ_DIR			= ./obj

# the library name
PRJ_NAME	 	= bms

# version information
MAJOR_VER		= 2
MINOR_VER		= 0
VER				= $(MAJOR_VER).$(MINOR_VER)

# defines to set
DEFS			= -DMSYS_LINUX -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -DMSYS_UNIX_LARGEFILE

# set objects
OBJS          	= \
			$(OBJ_DIR)/BMS.o \
			$(OBJ_DIR)/BMS360.o \
			$(OBJ_DIR)/UBMS.o \
			$(OBJ_DIR)/UBMS360.o \
						

LIBS				= -lpthread

DEBUG_LIBS			=
RELEASE_LIBS		=

STAT_LIBS			=
DYN_LIBS			=

# the libraries to link with
STAT_DEBUG_LIBS		= 
STAT_RELEASE_LIBS	= 
DYN_DEBUG_LIBS		=
DYN_RELEASE_LIBS	=

# name of the base makefile
MAKE_FILE_NAME		= ../../makefile.base

# include the base makefile
include $(MAKE_FILE_NAME)
