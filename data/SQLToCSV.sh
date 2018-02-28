# !/bin/bash
# This script is changing SQL files to csv file, the name of the file is
# the same as the table name in the database
# The program takes one argument in which is the .sql file, will return
# several .csv files.
# Note that the quote char is '@' instead of normal "

while IFS='' read -r line || [[ -n $line ]]; do
	if [[ $line == "CREATE TABLE"* ]]; then
		tmp=${line#*\"}
		filename=${tmp%%\"*}.csv
		if [ -e $filename ]; then
			rm $filename 2> /dev/null
		fi
		
		tmp=${line#*(}
		features=${tmp%)*}
		echo CREATE csvfile "$filename"
		echo $features | awk -F\" '{for (i=2; i <= NF; i+= 2) {print $i}}' | paste -sd"," > $filename
	elif [[ $line == "INSERT INTO"* ]]; then
		tmp=${line#*(}
		data=${tmp%)*}
		tmp=${data//,\'/,\@}
		tmp=${tmp//\',/\@,}
		echo ${tmp/%\'/\@} >> $filename
	fi
done < $1

