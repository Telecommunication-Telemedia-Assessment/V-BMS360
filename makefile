### enforce 32-bit build : 1=yes, 0=no
M32?= 0
export M32

GPU_MODE?=1
export GPU_MODE


all: 
	$(MAKE) -C lib/libgnomonic
	$(MAKE) -C lib/libbms
	$(MAKE) -C model
	$(MAKE) -C prior

libs:
	$(MAKE) -C lib/libgnomonic
	$(MAKE) -C lib/libbms

gpu:
	$(MAKE) -C lib/libgnomonic
	$(MAKE) -C lib/libbms
	$(MAKE) -C model 			GPU_MODE=1

cpu:
	$(MAKE) -C lib/libgnomonic
	$(MAKE) -C lib/libbms
	$(MAKE) -C model 			GPU_MODE=0

prior:
	$(MAKE) -C prior

clean :
	$(MAKE) -C lib/libgnomonic clean
	$(MAKE) -C lib/libbms clean
	$(MAKE) -C model clean
	$(MAKE) -C prior clean



