# 该文件放在项目的根目录中
BUILD_DIR = ./build
DEST_DIR = ./bin
DEST_EXE_NAME = main

all: build
	cd $(BUILD_DIR);make -j
run:
	cd $(DEST_DIR); ./$(DEST_EXE_NAME)
clean:
	rm $(BUILD_DIR) -rf
build:
	mkdir $(BUILD_DIR);cd $(BUILD_DIR);cmake ..
miniclean:
	cd $(BUILD_DIR);  rm -rf CMakeFiles;rm -rf CMakeCache.txt;
# make clean //清空build文件夹
# make build //在同级目录下新建build目录，生成makefile
# make all/make //编译成可执行文件，在bin文件夹
# make run //进入bin文件夹，运行可执行文件
# make miniclean // 只删除build中的CMakeFiles和CMakeCache.txt
