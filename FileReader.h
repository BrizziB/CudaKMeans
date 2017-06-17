#pragma once
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
class FileReader
{

private:
	char filePath;


public:
	FileReader();
	virtual ~FileReader();
	void readFile(std::string path, std::vector<Point>* points);
	void readFile(std::string path, std::vector<Centroid>* points);
};

