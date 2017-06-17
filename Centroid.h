#pragma once
#include <iostream>
#include <numeric>
#include <string>
#include <fstream>
#include <regex>
#include <stdlib.h>
#include "Point.h"

class Centroid
{
public:

	int ID;
	int numAttributes;
	double* attributes;
	int pointsLength;
	Point* centroidPoints = new Point[8];

	Centroid(){

	}

	Centroid(int id, int numAttribs, double* attribs){
		ID = id;
		attributes = attribs;
		numAttributes = numAttribs;
		pointsLength = 0;
		centroidPoints = {};
		
	}

	virtual ~Centroid()
	{
	}
};

