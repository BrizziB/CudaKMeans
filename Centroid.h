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
	float* attributes;
	int pointsLength;

	Centroid(){

	}

	Centroid(int id, int numAttribs, float* attribs){
		ID = id;
		attributes = attribs;
		numAttributes = numAttribs;
		pointsLength = 0;
		
	}

	virtual ~Centroid()
	{
	}
};

