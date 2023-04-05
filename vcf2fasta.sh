#!/bin/bash
#


while [ "$1" != "" ]; do
    case $1 in
        --vcf_file )        shift
                            vcf_file=$1
                            ;;
        --ref_fasta )       shift
                            ref_fasta=$1
                            ;;
        --results_dir )     shift
                            results_dir=$1
                            ;;
        --output_name )     shift
                            output_name=$1
                            ;;
    esac
    shift
done

mkdir -p $results_dir

## this is cluster specific; too bad! delete the line if u don't need it
ml biology bcftools 

sbatch -c 8 -J "vcf2fa_${output_name}" -o "${results_dir}/%A_vcf2fa_${output_name}" --mem=128G -p cgawad --time=5-00:00:00 \
    --wrap="cat $ref_fasta | bcftools consensus $vcf_path > $results_dir/$output_name"
