#!/bin/bash

for n in {3..8}
do
	filename='Celiac_n_'$n'_trimming_2_1'
	echo '~~~~n='$n
	python /media/miri-o/Documents/python_scripts/generate_model.py -f /media/miri-o/Documents/fasta_files/CDR3_celiac_db.fa -o /media/miri-o/Documents/Immune2vec/trained_models/$filename -n $n -t '(2,1)'
	echo 'Done!'
done

