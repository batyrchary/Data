import os
import csv



abstract=["abstract"]
introduction=["introduction","Introduction"]
related=["Related works","Background","Related Work","Related Works"]	
methodology=["methodology","Methodology","Methods"]
results=["Results","results","Evaluation","evaluation","Experimental Evaluation"]
conclusion=["Conclusion","conclusion","Conclusions"]

def check_Section(line):
	sectionChanged=False
	section=6 #others

	for a in abstract:
		if (a in line) and (5<len(line)<30):
			section=0
			sectionChanged=True

	for a in introduction:
		if (a in line) and (5<len(line)<30):
			section=1
			sectionChanged=True

	for a in related:
		if (a in line) and (5<len(line)<30):
			section=2
			sectionChanged=True

	for a in methodology:
		if (a in line) and (5<len(line)<30):
			section=3
			sectionChanged=True

	for a in results:
		if (a in line) and (5<len(line)<30):
			section=4
			sectionChanged=True

	for a in conclusion:
		if (a in line) and (5<len(line)<30):
			section=5
			sectionChanged=True

	return [section,sectionChanged]


if __name__ == '__main__':

	#abstract,introduction,related,methodology,results,conclusion,others
	sections=[[],[],[],[],[],[],[]]
	sectionTitle=["abstract","introduction","related","methodology","results","conclusion","rest"]

	dirlist = os.listdir("./raw")

	for d in dirlist:
		
		path="./raw/"+d
		num_lines = sum(1 for line in open(path))

		#print num_lines
		if num_lines<100:
			continue

		section=6 #others
		
		file=open(path)
		pathU="./cleaned/"+str(d).split(".txt")[0]+"U.txt"
		filew=open(pathU,"w+")

		for line in file:

			line=line.strip("\n\r")
			
			if(len(line)==0):
				continue
			
			sec_changed=check_Section(line)

			sectionChanged=sec_changed[1]

			if (sectionChanged==True):
				section=sec_changed[0]
				continue

			sections[section].append(line)

		index=0
		for s in sections:
			filew.write(sectionTitle[index]+"\n")
			index=index+1
			for sentence in s:
				filew.write(sentence)

			filew.write("\n")
		filew.close()


	

			