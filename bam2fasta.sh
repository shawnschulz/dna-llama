#!/bin/bash
#

ml biology samtools

dir=$1
fasta_dir=$2

# you'll need to set this directory to wear seqtk is installed or export path
seqtk_path='/home/groups/cgawad/conda_for_pf_notebook/seqtk/seqtk'
export PATH=$seqtk_path:$PATH

mkdir -p $fasta_dir

for file in $dir/*.bam; do
      file_name=$(basename ${file})
      fastq_file_name=$fasta_dir/${file_name%.bam}.fasta
      if [[ -n $fastq_file_name ]] | [[ ! -s $fastq_file_name ]]; then
          ls -lhtr $fastq_file_name
          echo $fastq_file_name
          echo $( ! -s ${fastq_file_name} )
          sbatch -c 2 -J "bam2qf_${file_name}" -o "${dir}/%A_bam2qf_${file_name}" --mem=32G -p cgawad --time=24:00:00 --wrap="samtools bam2fq $file | $seqtk_path seq -A > $fastq_file_name"
      fi
done