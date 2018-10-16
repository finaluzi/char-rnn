for ((i=660;i<1000;i+=33)) 
do
	echo 0.$i 。
	th sample.lua $1 -temperature 0.$i -length $2 -primetext $3 -gpuid $4 -seed $5
done
