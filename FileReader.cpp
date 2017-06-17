#include <stdio.h>
#include <iostream>
#include <numeric>
#include <string>
#include <fstream>
#include <regex>
#include <stdlib.h>
#include "FileReader.h"
#include "Point.h"
#include "Centroid.h"

using namespace std;

FileReader::FileReader()
{
}


FileReader::~FileReader()
{
}

void FileReader::readFile(string path, vector<Point>* points){

	ifstream datasetFile(path);
	string line;
	string output;
	//datasetFile.open;
	if (datasetFile.is_open()){
		cout << "\nreading points";
		cout.flush();
		regex rgx("\\b\\d[\\d,.]+[\\d,.]\\b|\\d+");
		smatch subMatch;
		vector<double> attributes;
		int index = 0;
		while (getline(datasetFile, line)){
			cout << "\n linea: "<<index;
			attributes.clear();
			int numAttribs = 0;
			while (regex_search(line, subMatch, rgx)){
				for (auto x : subMatch) attributes.push_back(stod(x));//registra x come double in entry
				numAttribs++;
				line = subMatch.suffix().str();
			}
			index++;
			
			double* attribs;
			attribs = (double*)malloc(sizeof(double)*numAttribs);
			
			for (int i = 0; i < numAttribs; i++){
				attribs[i] = attributes.at(i);
			}

			Point *newPoint = new Point(index, numAttribs, attribs);
			points->push_back(*newPoint);

		}
		datasetFile.close();
	}
}
void FileReader::readFile(string path, vector<Centroid>* points){

	ifstream datasetFile(path);
	string line;
	string output;
	//datasetFile.open;
	if (datasetFile.is_open()){
		cout << "\nreading centroids";
		cout.flush();
		regex rgx("\\b\\d[\\d,.]+[\\d,.]\\b|\\d+");
		smatch subMatch;
		vector<double> attributes;
		int index = 0;
		while (getline(datasetFile, line)){
			attributes.clear();
			int numAttribs = 0;
			while (regex_search(line, subMatch, rgx)){
				for (auto x : subMatch) attributes.push_back(stod(x));//registra x come double in entry
				numAttribs++;
				line = subMatch.suffix().str();
			}
			index++;

			
			double* attribs;
			attribs = (double*)malloc(sizeof(double)*numAttribs);
			for (int i = 0; i < numAttribs; i++){
				attribs[i] = attributes.at(i);
			}
			Centroid *newPoint = new Centroid(index, numAttribs,  attribs);
			points->push_back(*newPoint);

		}
		datasetFile.close();
	}
}