# -*- MakeFile -*-
all:
	g++ src/*.cpp -I. -Iinclude -o build/app

build:
	g++ src/*.cpp -I. -Iinclude -o build/app
	./build/app
