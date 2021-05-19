#!/usr/bin/env bash

VENVNAME=artist_classifier_venv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME