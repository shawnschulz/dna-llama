#!/bin/bash
#

sbatch=0
while [ "$1" != "" ]; do
    case $1 in
        --vcf_path )        shift
                            vcf_path=$1
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
        --sbatch )          shift
                            sbatch=$1
                            ;;
    esac
    shift
done

mkdir -p $results_dir

## this is cluster specific; too bad! delete the line if u don't need it
ml biology bcftools samtools

## this was a workaround for making the .tbi file but it didn't work
### later please make some shell commands that actually bgzip and tabix a vcf file properly lol

#unzip ${vcf_path}
#cat ${vcf_path%.gz} | \
#awk '$1 ~ /^#/ {print $0;next} {print $0 | "sort -k1,1 -k2,2n"}' | \
#bgzip --threads 20 > ${vcf_path}
#tabix ${vcf_path}

if [ $sbatch == '1' ]; then
   
    sbatch -c 8 -J "vcf2fa_${output_name}" -o "${results_dir}/%A_vcf2fa_${output_name}" --mem=128G -p cgawad --time=5-00:00:00 \
        --wrap="cat $ref_fasta | bcftools consensus $vcf_path > $results_dir/$output_name"
else
    cat $ref_fasta | bcftools consensus $vcf_path > $results_dir/$output_name
fi

##remove header lines
sed '/^>/d' $results_dir/$output_name

##actually want to keep the N's they have an encoding
#sed -i -e 's/N//g' $results_dir/$output_name
`
## make the file just a bigass string, i think this is unecessary tho since the encoder from enformer pytorch can use a sequence of strings
#tr -d '\n' < yourfile.txt
