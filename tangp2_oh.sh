for ((i=650;i<1000;i+=80)) 
do
	echo 0.$i
	th sample_oneHot.lua $1 -temperature 0.$i -length $2 -primetext $3 -gpuid $4 -seed $5
done
