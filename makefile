# makefile for ML Assignment 2
# CSC3022F 2021
#	Author: WNGJIA001

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv

run:
	. venv/bin/activate; python3 Example.py

clean:
	rm -rf venv
	rm -rf *.gif
	find . -iname "*.pyc" -delete
