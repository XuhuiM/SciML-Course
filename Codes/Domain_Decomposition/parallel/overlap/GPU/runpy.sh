#!/bin/bash
torchrun --standalone --nproc_per_node=2 ode_overlap.py
