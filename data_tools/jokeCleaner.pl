#!/usr/bin/perl

use strict;
use warnings;


my $myPDF = $ARGV[0] or die "file input not selected!\n"; #input argument, bulk text file from pdf


open('FILE',"< $myPDF") or die "error creating output file" ; #open the fd or die

open('STOPW',"< stopwords.txt") or die "error creating output file" ; #open the fd or die

######
my @stopWords = <STOPW>;  #slurp the whole thing into memory 
######
close(STOPW); #closing this fd quickly 
######




my $currentLine; 
my @currentStack;
my @bufferStack; 

my $string = "";
my @mainFileArray; #main array
my $forLoopCount;

my $i = 0; 

sub printAllLines(@); 




while(<FILE>)
{
	chomp($_);
	
	if($_ eq "#")
	{
		
		$forLoopCount = @bufferStack; 
		#~ print $forLoopCount; #just for debug
		
		#broke....
		#@bufferStack =~ s/@stopwords//g;


			for ($i = 0; $i < $forLoopCount; $i++)
			{
				#print 4; 
				pop(@bufferStack); #pop off all of the values 
			
			}
			
		print @bufferStack; 
			exit 0; 
		
		
	}
	else
	{		#print $_ . "\n"; 
			push(@bufferStack,$_); 
	}
	
	
}







#grab the congress number from records. and output a csv of congress number and file name 

# This tool was used to fix congress numbers not being in csv file

close(FILE); 


sub printAllLines(@) 
{
	#@_; 
	my $iter = 0; 
	
	for ($iter = 0; $iter < @_; $iter++	)
	{
		print $_[$iter] . "\n";
		
	}
}

exit 0; #exit out of the script right here 
