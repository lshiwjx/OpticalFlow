//
// Created by lshi on 3/16/18.
//

#ifndef OPTICALFLOW_MKDIR_H
#define OPTICALFLOW_MKDIR_H

#include <iostream>
#include <string>
#include <sys/stat.h> // stat
#include <errno.h>    // errno, ENOENT, EEXIST
#if defined(_WIN32)
#include <direct.h>   // _mkdir
#endif
bool makePath(const std::string& path);

#endif //OPTICALFLOW_MKDIR_H
