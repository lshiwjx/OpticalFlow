//
// Created by lshi on 3/1/18.
//

#ifndef OPTICALFLOW_LIST_FILE_H
#define OPTICALFLOW_LIST_FILE_H
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "vector"
#include <string.h>
#include "algorithm"

#ifdef linux
#include <unistd.h>
#include <dirent.h>
#endif
#ifdef WIN32
#include <direct.h>
#include <io.h>
#endif

using namespace std;
vector<string> getFiles(string cate_dir);


#endif //OPTICALFLOW_LIST_FILE_H
