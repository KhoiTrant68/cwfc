# Canonical reproduce commands. Override DATA_ROOT on Kaggle if needed:
#   make g1 DATA_ROOT=/kaggle/input/datasets/fedesoriano/cifar100
DATA_ROOT ?= /kaggle/input/datasets/fedesoriano/cifar100
PY ?= python

.PHONY: help smoke selftest g1 g1-full g4 g4-full toy dp-sandbox figs

help:
	@echo "Targets:"
	@echo "  smoke      quick synth plumbing check for g1_pilot + g4_wflow (CPU-ok)"
	@echo "  selftest   wflow_endpoint inversion self-test (torch only)"
	@echo "  toy        derisk_cot toy 2-D extended-cost frontier (Q1 + Q2)"
	@echo "  g1         G1 pilot, potential loss, weak AE + K=4 (the decisive run)"
	@echo "  g1-full    G1 robustness: 3 embeds x 3 seeds"
	@echo "  g4         G4 conditional-flow lambda frontier, seed 0"
	@echo "  g4-full    G4 hardened: 3 seeds + conditional-MMD"
	@echo "  figs       render frontier + D-P plane figures from results/ JSONs"

smoke:
	$(PY) g1_pilot.py --smoke
	$(PY) g4_wflow.py --dataset synth --H 32 --N 96 --npool 300 --knn 8 \
	  --steps 60 --ae_steps 100 --n_steps 8 --n_draws 4 --lams 0 0.5 1 --device cpu

selftest:
	$(PY) wflow_endpoint.py

toy:
	$(PY) derisk_cot.py --q both --N 1024 --steps 4000

g1:
	$(PY) g1_pilot.py --dataset cifar100 --data_root $(DATA_ROOT) \
	  --H 32 --mode both --N 384 --steps 2000 --ae_steps 3000 \
	  --ae_qscale 0.5 --ae_zc 2 --K 4 --eps 0.02 \
	  --seeds 0 --etas 0 0.3 3 30 --embeds pool --loss potential \
	  --out results/g1_potential.json

g1-full:
	$(PY) g1_pilot.py --dataset cifar100 --data_root $(DATA_ROOT) \
	  --H 32 --mode train --N 384 --steps 2000 --ae_steps 3000 \
	  --ae_qscale 0.5 --ae_zc 2 --K 4 --eps 0.02 \
	  --seeds 0 1 2 --etas 0 0.3 3 30 --embeds raw proj8 pool --loss potential \
	  --out results/g1_potential_full.json

g4:
	$(PY) g4_wflow.py --dataset cifar100 --data_root $(DATA_ROOT) \
	  --H 32 --N 256 --npool 2000 --knn 16 --steps 1500 --ae_steps 3000 \
	  --ae_zc 2 --ae_qscale 0.5 --n_steps 20 --n_draws 8 \
	  --lams 0 0.25 0.5 0.75 1 --seeds 0 --embed pool --out results/g4_smoke.json

g4-full:
	$(PY) g4_wflow.py --dataset cifar100 --data_root $(DATA_ROOT) \
	  --H 32 --N 256 --npool 2000 --knn 16 --steps 1500 --ae_steps 3000 \
	  --ae_zc 2 --ae_qscale 0.5 --n_steps 20 --n_draws 8 \
	  --lams 0 0.25 0.5 0.75 1 --seeds 0 1 2 --embed pool --out results/g4_full.json \
	  --save_samples figs

figs:
	$(PY) viz.py --g1 results/g1_potential_full.json \
	  --g4 results/g4_full.json --out figs
