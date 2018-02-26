#!/usr/bin/perl

use strict;
use warnings;


my $myPDF = $ARGV[0] or die "file input not selected!\n"; #input argument, bulk text file from pdf


open('FILE',"< $myPDF") or die "error creating output file" ; #open the fd or die


my $currentLine; 
my @currentStack;
my @bufferStack; 

my $string = "";


while(<FILE>)
{
	chomp($_);
	if($_ eq "#")
	{
		
		
		
		
	}
	else
	{
		print $_ . "\n"; 
		
	}
	
	
}



my $endString = "The case is submitted."; #when to stop inside of congressional record parsing

my @mainFileArray; #main array
my $outFile = "list.csv";  #output file name

open('OFILE',"> $outFile") or die "error creating output file" ; #open the fd or die




#grab the congress number from records. and output a csv of congress number and file name 

# This tool was used to fix congress numbers not being in csv file

