# the SOURCE definiton lets you move your makefile to another position
CONFIG 				= CONSOLE

# set directories to your wanted values
SRC_DIR				= ./src/
INC_DIR				= ./src/
LIB_DIR				= 
BIN_DIR				= ../bin/

SRC_DIR1		=
SRC_DIR2		=
SRC_DIR3		=
SRC_DIR4		=

USER_INC_DIRS	= -I$(SRC_DIR) \
				-I/usr/local/opt/boost/include \
			 	-I/usr/local/opt/eigen/include/eigen3 \
                -I../lib/libfplus/include \
                -I../lib/libjson/include \
                -I../lib/libfdeep/include \
                -I../lib/libbms/src \
                -I../lib/libgnomonic/src -I../lib/libgnomonic/lib/libinter/src \

USER_LIB_DIRS	= /usr/local/opt/boost/lib \
		 		  -L../lib/libgnomonic/bin -L../lib/libgnomonic/lib/libinter/bin \
		 		  -L../lib/libgnomonic/bin \
		 		  -L../lib/libbms/lib \





# intermediate directory for object files
OBJ_DIR				= ./obj/

# set executable name
PRJ_NAME			= salient

# defines to set
DEFS				= 

# set objects
OBJS          		= \
						$(OBJ_DIR)/common-method.o \
						$(OBJ_DIR)/FlowClassifier.o \
						$(OBJ_DIR)/FlowGrabber.o \
						$(OBJ_DIR)/main.o \
						$(OBJ_DIR)/FlowIO.o \
						$(OBJ_DIR)/MotionSourceFeatureMap.o \
						$(OBJ_DIR)/ObjectMotionFeatureMap.o \
						$(OBJ_DIR)/ImageFeatureMap.o \
						$(OBJ_DIR)/SalientFeatureMap.o \
						$(OBJ_DIR)/MotionFeatureMap.o \
						$(OBJ_DIR)/SalientFeatureFactory.o \
						$(OBJ_DIR)/Saliency360.o \
						$(OBJ_DIR)/BMSSaliency.o \
						$(OBJ_DIR)/EquatorialPrior.o \
						$(OBJ_DIR)/AdaptiveMotionFeatureMap.o \
						$(OBJ_DIR)/TrackedObjectFeatureMap.o \
						$(OBJ_DIR)/PedestrianDetectFeatureMap.o \
						$(OBJ_DIR)/TemporalPrior.o \
						$(OBJ_DIR)/SpatioTemporalFeatureMap.o \

						


# set libs to link with
LIBS				= -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_ximgproc -lopencv_video -lopencv_tracking  \
					  -lboost_program_options -lboost_exception -lboost_thread-mt -lboost_system -lboost_regex-mt \

DEBUG_LIBS			= -lgnomonicd -linterd
RELEASE_LIBS		= -lgnomonic -linter

STAT_LIBS			= -lpthread
DYN_LIBS			=


DYN_DEBUG_LIBS		= -lbmsd
DYN_DEBUG_PREREQS	=
STAT_DEBUG_LIBS		= -lbmsStaticd
STAT_DEBUG_PREREQS	=

DYN_RELEASE_LIBS	= -lbms
DYN_RELEASE_PREREQS	= 
STAT_RELEASE_LIBS	= -lbmsStatic
STAT_RELEASE_PREREQS= 



ifeq ($(GPU_MODE), 1)
DEFS 			+= -DGPU_MODE=1
USER_INC_DIRS	+= -I/usr/local/opencv_cuda/include
USER_LIB_DIRS 	+= -L/usr/local/opencv_cuda/lib
LIBS 			+= -lopencv_cudaoptflow -lopencv_cudaarithm
LIBS 			+= -rpath /usr/local/opencv_cuda/lib/
else
USER_INC_DIRS	+= -I/usr/local/opt/opencv/include
USER_LIB_DIRS 	+= -L/usr/local/opt/opencv/lib 
LIBS 			+= -lopencv_optflow
endif




# name of the base makefile
MAKE_FILE_NAME		= ../makefile.base

# include the base makefile
include $(MAKE_FILE_NAME)