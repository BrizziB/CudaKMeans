#pragma once
#include <iostream>
#include <numeric>
#include <string>
#include <fstream>
#include <regex>
#include <stdlib.h>
class Point
{
public:

	int ID;
	int numAttributes;
	float* attributes;

	Point(){

	}

	Point(int id, int numAttribs, float* attribs){
		ID = id;
		numAttributes = numAttribs;
		attributes = attribs;
	}

	virtual ~Point(){
	}

};

