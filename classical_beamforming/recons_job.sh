#!/bin/bash

# SLURM options:

#SBATCH --job-name=heron_classicalbf_%j_   # Nom du job
#SBATCH --output=../../logs/heron_classicalbf_%j.log   # Standard output et error log

#SBATCH --partition=htc               # Choix de partition (htc par défaut)

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=15000                   # Mémoire en MB par défaut

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --time=0-10:00:00             # Délai max = 7 jours

#SBATCH --mail-user=arsene.ferriere@lpnhe.in2p3.fr   # Où envoyer l'e-mail
#SBATCH --mail-type=END,FAIL          # �~Ivénements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles

# Commandes à soumettre :

source /sps/grand/aferrier/ccenvtest/bin/activate
python beamforming_recons.py --first-event 336 --last-event 500