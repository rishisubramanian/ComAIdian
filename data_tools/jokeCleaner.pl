#!/usr/bin/perl

use strict;
use warnings;

my $text = qq{Launch the application like so "jokeCleaner.pl jokeData.txt > outputJoke.csv"} . "\n\n\n";
my $myTxtFile = $ARGV[0] or die $text; #input argument, bulk text file from pdf

open('FILE',"< $myTxtFile") or die "error creating output file" ; #open the fd or die





my $currentLine; 
my @currentStack;
my @bufferStack; 

my $string = "";
my @mainFileArray; #main array
my $forLoopCount;
my @doubleBufferStack;
my $i = 0; 


my @stopList = ("a",
"about",
"above",
"after",
"again",
"against",
"all",
"am",
"an",
"and",
"any",
"are",
"aren't",
"as",
"at",
"be",
"because",
"been",
"before",
"being",
"below",
"between",
"both",
"but",
"by",
"can't",
"cannot",
"could",
"couldn't",
"did",
"didn't",
"do",
"does",
"doesn't",
"doing",
"don't",
"down",
"during",
"each",
"few",
"for",
"from",
"further",
"had",
"hadn't",
"has",
"hasn't",
"have",
"haven't",
"having",
"he",
"he'd",
"he'll",
"he's",
"her",
"here",
"here's",
"hers",
"herself",
"him",
"himself",
"his",
"how",
"how's",
"i",
"i'd",
"i'll",
"i'm",
"i've",
"if",
"in",
"into",
"is",
"isn't",
"it",
"it's",
"its",
"itself",
"let's",
"me",
"more",
"most",
"mustn't",
"my",
"myself",
"no",
"nor",
"not",
"of",
"off",
"on",
"once",
"only",
"or",
"other",
"ought",
"our",
"ours",
"ourselves",
"out",
"over",
"own",
"same",
"shan't",
"she",
"she'd",
"she'll",
"she's",
"should",
"shouldn't",
"so",
"some",
"such",
"than",
"that",
"that's",
"the",
"their",
"theirs",
"them",
"themselves",
"then",
"there",
"there's",
"these",
"they",
"they'd",
"they'll",
"they're",
"they've",
"this",
"those",
"through",
"to",
"too",
"under",
"until",
"up",
"very",
"was",
"wasn't",
"we",
"we'd",
"we'll",
"we're",
"we've",
"were",
"weren't",
"what",
"what's",
"when",
"when's",
"where",
"where's",
"which",
"while",
"who",
"who's",
"whom",
"why",
"why's",
"with",
"won't",
"would",
"wouldn't",
"you",
"you'd",
"you'll",
"you're",
"you've",
"your",
"yours",
"yourself",
"yourselves",
"a:", "q:","q", ":");

my ($rx) = map qr/(?:$_)/, join "|", map qr/\b\Q$_\E\b/, @stopList;

#~ my $line = "the good bear a good one"; #debugging it
#~ $line =~ ;

print qq{"original","clean"}; print "\n"; 


while(<FILE>)
{
	chomp; # remove newlines
    s/^\s+//;  # remove leading whitespace
    s/\s+$//; # remove trailing whitespace
    next unless length; # next rec unless anything left

	if($_ eq "#")
	{
		
		#~ $forLoopCount = @bufferStack; 
		#~ print $forLoopCount; #just for debug
		#~ @doubleBufferStack = lc(@bufferStack);
		
		s/$rx//g foreach (@doubleBufferStack);		
		s/""//g foreach (@doubleBufferStack);		
		s/''//g foreach (@doubleBufferStack);		
		s/://g foreach (@doubleBufferStack);	
		s/a://g foreach (@doubleBufferStack);	
		s/q://g foreach (@doubleBufferStack);	
		
		print qq{"@bufferStack","@doubleBufferStack"}; 
		#~ print ",";
		#~ print ;
		print "\n";
		
		@doubleBufferStack=(); 
		@bufferStack=(); 
		
		
		
	}
	else
	{		
			push(@bufferStack,$_ . " ");
			push(@doubleBufferStack,lc($_ . " ")); 
	}
	
	
}




close(FILE); 



exit 0; 


